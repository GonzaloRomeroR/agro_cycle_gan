import random
from typing import Any

import torch

from .base_trainer import BaseTrainer


class BasicTrainer(BaseTrainer):
    def _set_training_params(self) -> None:
        self.lambda_cycle = 5
        self.lambda_identity = 10

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

                if iters == 0 and epoch == 0:
                    old_b_fake = b_fake.clone()
                    old_a_fake = a_fake.clone()

                # Discriminator
                disc_losses = {}
                for disc in [self.D_A, self.D_B]:
                    disc_name = "DISC_A" if disc == self.D_A else "DISC_B"
                    im_real = a_real if disc == self.D_A else b_real
                    im_fake = a_fake if disc == self.D_A else b_fake
                    old_im_fake = old_a_fake if disc == self.D_A else old_b_fake
                    D_losses = self.D_A_losses if disc == self.D_A else self.D_B_losses

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

                # Store results
                for losses in [gen_losses, disc_losses]:
                    for name, value in losses.items():
                        self.losses_epoch[name].append(value.item())

                # Update old image fake
                if iters == 0 and epoch == 0:
                    old_b_fake = b_fake.clone()
                    old_a_fake = a_fake.clone()
                elif old_b_fake.shape[0] == self.bs * 5 and b_fake.shape[0] == self.bs:
                    rand_int = random.randint(5, 24)
                    old_b_fake[rand_int - 5 : rand_int] = b_fake.clone()
                    old_a_fake[rand_int - 5 : rand_int] = a_fake.clone()
                elif old_b_fake.shape[0] < 25:
                    old_b_fake = torch.cat((b_fake.clone(), old_b_fake))
                    old_a_fake = torch.cat((a_fake.clone(), old_a_fake))

                iters += 1

                # Print iteration results
                if iters % self.print_info == 0:
                    self._print_iter_info(epoch, i, gen_losses, disc_losses)

            iters = 0
            self._run_post_epoch(epoch)

