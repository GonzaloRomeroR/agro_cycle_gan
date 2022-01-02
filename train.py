import torch
import os
import random
import time

from datetime import datetime
from utils.file_utils import download_images, save_models, get_models
from utils.image_utils import datasets_get
from utils.plot_utils import plot_generator_images
from utils.report_utils import generate_report, generate_model_file, ParamsLogger
from utils.tensorboard_utils import create_models_tb, TensorboardHandler
from utils.sys_utils import get_device, suppress_qt_warnings, suppress_sklearn_errors
from utils.metrics_utils import FID, calculate_metrics
from utils.parser_utils import parse_arguments
from utils.train_utils import generate_images_cycle


def setup(cmd_args):
    print("Performing setup")

    # Get images
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

    # Get datasets from both domains
    batch_size = cmd_args.batch_size
    train_images_A, train_images_B, test_images_A, test_images_B = datasets_get(
        dataset_name, im_size[1:], batch_size
    )

    # Create or load models for training
    G_A2B, G_B2A, D_A, D_B = get_models(dataset_name, device, cmd_args.load_models)

    # Set tesorboard functions
    if cmd_args.tensorboard:
        images, _ = next(iter(train_images_A))
        create_models_tb(G_A2B, G_B2A, D_A, D_B, images.to(device))

    # Create metrics object
    metrics = FID() if cmd_args.metrics else None

    # Generate file with the model information
    generate_model_file(G_A2B, G_B2A, D_A, D_B, size=im_size)

    # Define params for report
    params_logger = ParamsLogger()
    domains = cmd_args.use_dataset.split("2")
    log_params = {
        "date": datetime.now(),
        "dataset": cmd_args.use_dataset,
        "image_size": im_size,
        "batch_size": batch_size,
        "configured_epochs": cmd_args.num_epochs,
    }
    if len(domains) == 2:
        # The dataset name format is domainA2domainB
        log_params["domain_A"] = domains[0]
        log_params["domain_B"] = domains[1]

    for name, model in {"G_A2B": G_A2B, "G_B2A": G_B2A, "D_A": D_A, "D_B": D_B}.items():
        log_params[f"model_name_{name}"] = model.__class__.__name__
        log_params[f"learning_rate_{name}"] = model.lr
    params_logger.set_params(log_params)

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
        params_logger=params_logger,
        metrics=metrics,
        im_size=im_size,
    )

    # Generate report
    generate_report(losses)


def train(
    G_A2B,
    G_B2A,
    D_A,
    D_B,
    path,
    images_A,
    images_B,
    dataset_name,
    device,
    test_images_A,
    test_images_B,
    bs=5,
    num_epochs=20,
    plot_epochs=1,
    print_info=3,
    params_logger=None,
    metrics=None,
    im_size=(64, 64),
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
    :param dataset_name: name of the models
    :type dataset_name: str
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
    :param params_logger: logger to store the training information, defaults to None
    :type params_logger: `ParamsLogger`, optional
    :param metrics: logger to store the training information, defaults to None
    :type metrics: `Metrics`, optional
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
        "GEN_TOTAL",
        "DISC_A",
        "DISC_B",
    ]

    writer = TensorboardHandler("./runs/Losses")

    losses_epoch = {key: [] for key in losses_names}
    losses_total = {key: [] for key in losses_names}

    start = time.perf_counter()

    for epoch in range(0, num_epochs):
        print("\n" + "=" * 20)
        print(f"Epoch: [{epoch}/{num_epochs}]")

        for i, (data_A, data_B) in enumerate(zip(images_A, images_B), 1):
            # Set model input
            a_real = data_A[0].to(device)
            b_real = data_B[0].to(device)

            # Generate images
            a_fake, b_fake, a_recon, b_recon = generate_images_cycle(
                a_real, b_real, G_A2B, G_B2A
            )

            if iters == 0 and epoch == 0:
                old_b_fake = b_fake.clone()
                old_a_fake = a_fake.clone()

            # Discriminator
            disc_losses = {}
            for disc in [D_A, D_B]:
                disc_name = "DISC_A" if disc == D_A else "DISC_B"
                im_real = a_real if disc == D_A else b_real
                im_fake = a_fake if disc == D_A else b_fake
                old_im_fake = old_a_fake if disc == D_A else old_b_fake
                D_losses = D_A_losses if disc == D_A else D_B_losses

                disc.optimizer.zero_grad()
                if (iters > 0 or epoch > 0) and iters % 3 == 0:
                    rand_int = random.randint(5, old_im_fake.shape[0] - 1)
                    disc_loss = disc.get_loss(
                        disc(im_real),
                        disc(old_im_fake[rand_int - 5 : rand_int].detach()),
                    )
                else:
                    disc_loss = disc.get_loss(disc(im_real), disc(im_fake.detach()))

                D_losses.append(disc_loss.item())
                disc_loss.backward()
                disc.optimizer.step()
                disc_losses[disc_name] = disc_loss

            # Generator
            gen_losses = {}
            G_A2B.optimizer.zero_grad()
            G_B2A.optimizer.zero_grad()

            # Fool discriminator loss
            gen_losses["FDL_A2B"] = G_A2B.get_loss(D_B(b_fake))
            gen_losses["FDL_B2A"] = G_B2A.get_loss(D_A(a_fake))

            # Cycle consistency loss
            gen_losses["CL_A"] = G_B2A.cycle_criterion(a_recon, a_real) * 5
            gen_losses["CL_B"] = G_A2B.cycle_criterion(b_recon, b_real) * 5

            # Identity loss
            gen_losses["ID_B2A"] = G_B2A.cycle_criterion(G_B2A(a_real), a_real) * 10
            gen_losses["ID_A2B"] = G_A2B.cycle_criterion(G_A2B(b_real), b_real) * 10

            # Generator losses
            loss_G = sum(gen_losses.values())
            G_losses.append(loss_G)
            gen_losses["GEN_TOTAL"] = loss_G

            # Backward propagation
            loss_G.backward()

            # Optimisation step
            G_A2B.optimizer.step()
            G_B2A.optimizer.step()

            # Store results
            for losses in [gen_losses, disc_losses]:
                for name, value in losses.items():
                    losses_epoch[name].append(value.item())

            # Update old image fake
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

            # Print iteration results
            if iters % print_info == 0:
                info = f"Epoch [{epoch}/{num_epochs}] batch [{i}]"
                for name, loss in {**gen_losses, **disc_losses}.items():
                    info += f" {name}: {loss:.3f}"
                print(info)

        # Obtain total losses
        for key in losses_epoch.keys():
            loss_key = sum(losses_epoch[key]) / len(losses_epoch[key])
            losses_total[key].append(loss_key)
            writer.add_scalar(key, loss_key, epoch)

        losses_epoch = {key: [] for key in losses_names}

        iters = 0

        # Generate epoch information
        params_logger.params["final_epoch"] = epoch
        params_logger.params["time"] = time.perf_counter() - start

        params_logger.generate_params_file()
        save_models(path, dataset_name, G_A2B, G_B2A, D_A, D_B)
        if epoch % plot_epochs == 0:
            plot_generator_images(G_A2B, G_B2A, test_images_A, test_images_B, device)

        # Obtain metrics
        if metrics:
            score = calculate_metrics(metrics, dataset_name, im_size[1:])
            print(f"{metrics.name} score: {score}")

    end_time = time.perf_counter() - start
    print(f"Full trainig took {end_time} s to finish")

    return losses_total


def configure():
    suppress_sklearn_errors()
    suppress_qt_warnings()


if __name__ == "__main__":
    configure()
    cmd_args = parse_arguments()
    setup(cmd_args)

