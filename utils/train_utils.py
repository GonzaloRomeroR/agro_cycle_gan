def generate_images_cycle(a_real, b_real, G_A2B, G_B2A):
    """
    Create fake and reconstructed images.
    """
    b_fake = G_A2B(a_real)
    a_recon = G_B2A(b_fake)
    a_fake = G_B2A(b_real)
    b_recon = G_A2B(a_fake)
    return a_fake, b_fake, a_recon, b_recon
