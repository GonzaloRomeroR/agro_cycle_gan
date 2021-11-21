import torch
import argparse
import os
import random

from utils.file_utils import download_images, load_models, save_models, create_models
from utils.image_utils import get_datasets
from utils.plot_utils import plot_generator_images
from utils.report_utils import generate_report, generate_model_file
from utils.tensorboard_utils import create_models_tb, TensorboardHandler
from utils.sys_utils import get_device, suppress_qt_warnings


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def setup(cmd_args):
    print("Performing setup")
    device = get_device(debug=True)
    dataset_name = cmd_args.use_dataset
    if cmd_args.download_dataset:
        print(f"Downloading dataset {dataset_name}")
        download_images(dataset_name)

    # Set image size
    image_resize = cmd_args.image_resize
    if image_resize:
        im_size = tuple([3] + image_resize)
    else:
        im_size = (3, 64, 64)

    train_images_A, train_images_B = get_datasets(
        dataset_name=dataset_name,
        dataset="train",
        im_size=im_size[1:],
        batch_size=cmd_args.batch_size,
    )
    test_images_A, test_images_B = get_datasets(
        dataset_name=dataset_name,
        dataset="test",
        im_size=im_size[1:],
        batch_size=cmd_args.batch_size,
    )

    if cmd_args.load_models:
        G_A2B, G_B2A, D_A, D_B = load_models(f"./results/{dataset_name}", dataset_name)
    else:
        G_A2B, G_B2A, D_A, D_B = create_models(device)

    if cmd_args.tensorboard:
        images, _ = next(iter(train_images_A))
        create_models_tb(G_A2B, G_B2A, D_A, D_B, images.to(device))

    generate_model_file(G_A2B, G_B2A, D_A, D_B, size=im_size)

    losses = train(
        G_A2B,
        G_B2A,
        D_A,
        D_B,
        f"./results/{dataset_name}",
        train_images_A,
        train_images_B,
        dataset_name,
        num_epochs=cmd_args.num_epochs,
        device=device,
        test_images_A=test_images_A,
        test_images_B=test_images_B,
        bs=cmd_args.batch_size,
    )

    generate_report(losses)


def train(
    G_A2B,
    G_B2A,
    D_A,
    D_B,
    path,
    images_A,
    images_B,
    name,
    device,
    test_images_A,
    test_images_B,
    bs=5,
    num_epochs=20,
    plot_epochs=1,
    print_info=3,
):
    """Train the generator and the discriminator

    :param G_A2B: generator to transform images from A to B
    :type G_A2B: `Generator`
    :param G_B2A: generator to transform images from B to A
    :type G_B2A: `Generator`
    :param D_A: discriminator for images in the domain A
    :type D_A: `Discriminator`
    :param D_B: discriminator for images in the domain A
    :type D_B: `Discriminator`
    :param images_A: dataloader with images of the domain A
    :type images_A: ´DataLoader´
    :param images_B: dataloader with images of the domain B
    :type images_B: ´DataLoader´
    :param path: path to load and save the models
    :type path: str
    :param name: name of the models
    :type name: str
    :param device: pytorch device
    :type device: ´Device´
    :param bs: batch size, defaults to 5
    :type bs: int, optional
    :param num_epochs: number of epochs, defaults to 20
    :type num_epochs: int, optional
    :param plot_epochs: epochs before printing images, defaults to 5
    :type plot_epochs: int, optional
    :param print_info: batchs processed before printing losses, defaults to 3
    :type print_info: int, optional
    """
    print("Starting Training Loop...")
    iters = 0
    D_A_losses = []
    D_B_losses = []
    G_losses = []

    losses_names = [
        "FDL_A2B",
        "FDL_B2A",
        "CL_A",
        "CL_B",
        "ID_B2A",
        "ID_A2B",
        "gen",
        "disc_A",
        "disc_B",
    ]

    writer = TensorboardHandler("./runs/Losses")

    losses_epoch = {key: [] for key in losses_names}
    losses_total = {key: [] for key in losses_names}

    for epoch in range(0, num_epochs):
        print("\n" + "=" * 20)
        print(f"Epoch: [{epoch}/{num_epochs}]")
        for i, (data_A, data_B) in enumerate(zip(images_A, images_B), 1):
            # Set model input
            a_real = data_A[0].to(device)
            b_real = data_B[0].to(device)

            # Generate images
            b_fake = G_A2B(a_real)
            a_recon = G_B2A(b_fake)
            a_fake = G_B2A(b_real)
            b_recon = G_A2B(a_fake)

            if iters == 0 and epoch == 0:
                old_b_fake = b_fake.clone()
                old_a_fake = a_fake.clone()

            # Discriminator A
            D_A.optimizer.zero_grad()
            if (iters > 0 or epoch > 0) and iters % 3 == 0:
                rand_int = random.randint(5, old_a_fake.shape[0] - 1)
                D_loss_A = D_A.get_loss(
                    D_A(a_real), D_A(old_a_fake[rand_int - 5 : rand_int].detach())
                )
            else:
                D_loss_A = D_A.get_loss(D_A(a_real), D_A(a_fake.detach()))

            D_A_losses.append(D_loss_A.item())
            D_loss_A.backward()
            D_A.optimizer.step()

            # Discriminator B
            D_B.optimizer.zero_grad()
            if (iters > 0 or epoch > 0) and iters % 3 == 0:
                rand_int = random.randint(5, old_b_fake.shape[0] - 1)
                D_loss_B = D_B.get_loss(
                    D_B(b_real), D_B(old_b_fake[rand_int - 5 : rand_int].detach())
                )
            else:
                D_loss_B = D_B.get_loss(D_B(b_real), D_B(b_fake.detach()))

            D_B_losses.append(D_loss_B.item())
            D_loss_B.backward()
            D_B.optimizer.step()

            # Generator
            G_A2B.optimizer.zero_grad()
            G_B2A.optimizer.zero_grad()

            # Fool discriminator
            fool_disc_loss_A2B = G_A2B.get_loss(D_B(b_fake))
            fool_disc_loss_B2A = G_B2A.get_loss(D_A(a_fake))

            # Cycle consistency loss
            cycle_loss_A = G_B2A.cycle_criterion(a_recon, a_real) * 5
            cycle_loss_B = G_A2B.cycle_criterion(b_recon, b_real) * 5

            # Identity loss
            id_loss_B2A = G_B2A.cycle_criterion(G_B2A(a_real), a_real) * 10
            id_loss_A2B = G_A2B.cycle_criterion(G_A2B(b_real), b_real) * 10

            # Generator losses
            loss_G = (
                fool_disc_loss_A2B
                + fool_disc_loss_B2A
                + cycle_loss_A
                + cycle_loss_B
                + id_loss_B2A
                + id_loss_A2B
            )
            G_losses.append(loss_G)

            # Backward propagation
            loss_G.backward()

            # Optimisation step
            G_A2B.optimizer.step()
            G_B2A.optimizer.step()

            # Store results
            losses_epoch["FDL_A2B"].append(fool_disc_loss_A2B.item())
            losses_epoch["FDL_B2A"].append(fool_disc_loss_B2A.item())
            losses_epoch["CL_A"].append(cycle_loss_A.item())
            losses_epoch["CL_B"].append(cycle_loss_B.item())
            losses_epoch["ID_B2A"].append(id_loss_B2A.item())
            losses_epoch["ID_A2B"].append(id_loss_A2B.item())
            losses_epoch["disc_A"].append(D_loss_A.item())
            losses_epoch["disc_B"].append(D_loss_B.item())
            losses_epoch["gen"].append(loss_G.item())

            if iters == 0 and epoch == 0:
                old_b_fake = b_fake.clone()
                old_a_fake = a_fake.clone()
            elif old_b_fake.shape[0] == bs * 5 and b_fake.shape[0] == bs:
                rand_int = random.randint(5, 24)
                old_b_fake[rand_int - 5 : rand_int] = b_fake.clone()
                old_a_fake[rand_int - 5 : rand_int] = a_fake.clone()
            elif old_b_fake.shape[0] < 25:
                old_b_fake = torch.cat((b_fake.clone(), old_b_fake))
                old_a_fake = torch.cat((a_fake.clone(), old_a_fake))

            iters += 1
            if iters % print_info == 0:
                info = f"Epoch [{epoch}/{num_epochs}] batch [{i}] FDL_A2B {fool_disc_loss_A2B:.3f} "
                info += f"FDL_B2A {fool_disc_loss_B2A:.3f} CL_A {cycle_loss_A:.3f} "
                info += f"CL_B {cycle_loss_B:.3f} ID_B2A {id_loss_B2A:.3f} "
                info += f"ID_A2B {id_loss_A2B:.3f} Loss_D_A {D_loss_A:.3f} "
                info += f"Loss_D_B {D_loss_B:.3f} "
                print(info)

        for key in losses_epoch.keys():
            loss_key = sum(losses_epoch[key]) / len(losses_epoch[key])
            losses_total[key].append(loss_key)
            writer.add_scalar(key, loss_key, epoch)

        losses_epoch = {key: [] for key in losses_names}

        iters = 0
        save_models(path, name, G_A2B, G_B2A, D_A, D_B)
        if epoch % plot_epochs == 0:
            plot_generator_images(G_A2B, G_B2A, test_images_A, test_images_B, device)
    return losses_total


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train GANs to generate image synthetic datasets."
    )
    parser.add_argument(
        "use_dataset", type=str, help="name of the dataset to use", default=None,
    )
    parser.add_argument(
        "--download_dataset", help="download or obtain dataset", action="store_true"
    )
    parser.add_argument(
        "--image_resize",
        help="size of the image to resize",
        default=None,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--tensorboard", help="generate tensorboard files", action="store_true",
    )
    parser.add_argument(
        "--load_models",
        action="store_true",
        help="load the trained models from previous saves",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="batch size for the training process",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="number of epochs to train",
    )
    return parser.parse_args()


if __name__ == "__main__":
    suppress_qt_warnings()
    cmd_args = parse_arguments()
    setup(cmd_args)

