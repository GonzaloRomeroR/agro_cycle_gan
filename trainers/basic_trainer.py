from typing import Any
from .base_trainer import BaseTrainer


class BasicTrainer(BaseTrainer):
    def _set_training_params(self) -> None:
        self.lambda_cycle = 5
        self.lambda_identity = 10

    def get_discriminator_losses(self, a_real, b_real, a_fake, b_fake):

        disc_losses = {}
        for disc in [self.D_A, self.D_B]:
            disc_name = "DISC_A" if disc == self.D_A else "DISC_B"
            im_real = a_real if disc == self.D_A else b_real
            im_fake = a_fake if disc == self.D_A else b_fake
            D_losses = self.D_A_losses if disc == self.D_A else self.D_B_losses

            disc.optimizer.zero_grad()
            disc_loss = disc.get_loss(disc(im_real), disc(im_fake.detach()))

            D_losses.append(disc_loss.item())
            disc_loss.backward()
            disc.optimizer.step()
            disc_losses[disc_name] = disc_loss
        return disc_losses

    def get_generator_losses(self, a_real, b_real, a_fake, b_fake, a_recon, b_recon):
        # Generator
        gen_losses = {}
        self.G_A2B.optimizer.zero_grad()
        self.G_B2A.optimizer.zero_grad()

        # Fool discriminator loss
        gen_losses["FDL_A2B"] = self.G_A2B.get_loss(self.D_B(b_fake))
        gen_losses["FDL_B2A"] = self.G_B2A.get_loss(self.D_A(a_fake))

        # Cycle consistency loss
        gen_losses["CL_A"] = (
            self.G_B2A.cycle_criterion(a_recon, a_real) * self.lambda_cycle
        )
        gen_losses["CL_B"] = (
            self.G_A2B.cycle_criterion(b_recon, b_real) * self.lambda_cycle
        )

        # Identity loss
        gen_losses["ID_B2A"] = (
            self.G_B2A.cycle_criterion(self.G_B2A(a_real), a_real)
            * self.lambda_identity
        )
        gen_losses["ID_A2B"] = (
            self.G_A2B.cycle_criterion(self.G_A2B(b_real), b_real)
            * self.lambda_identity
        )

        # Generator losses
        loss_G: Any = sum(gen_losses.values())
        self.G_losses.append(loss_G)
        gen_losses["GEN_TOTAL"] = loss_G

        # Backward propagation
        loss_G.backward()

        # Optimisation step
        self.G_A2B.optimizer.step()
        self.G_B2A.optimizer.step()

        return gen_losses

    def _train_model(self) -> None:
        iters = 0
        for epoch in range(0, self.num_epochs):
            print("\n" + "=" * 20)
            print(f"Epoch: [{epoch}/{self.num_epochs}]")

            for i, (data_A, data_B) in enumerate(zip(self.images_A, self.images_B), 1):
                # Set model input
                a_real = data_A[0].to(self.device)
                b_real = data_B[0].to(self.device)

                # Generate images
                a_fake, b_fake, a_recon, b_recon = self.generate_images_cycle(
                    a_real, b_real, self.G_A2B, self.G_B2A
                )

                # Discriminator
                disc_losses = self.get_discriminator_losses(
                    a_real, b_real, a_fake, b_fake
                )

                gen_losses = self.get_generator_losses(
                    a_real, b_real, a_fake, b_fake, a_recon, b_recon
                )

                # Store results
                for losses in [gen_losses, disc_losses]:
                    for name, value in losses.items():
                        self.losses_epoch[name].append(value.item())

                iters += 1

                # Print iteration results
                if iters % self.print_info == 0:
                    self._print_iter_info(epoch, i, gen_losses, disc_losses)

            iters = 0
            self._run_post_epoch(epoch)
