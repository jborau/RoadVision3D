import wandb

class WandbLogger:
    def __init__(self, project, config=None, name=None, notes=None, tags=None):
        """
        Initialize the wandb logger.
        """
        self.run = wandb.init(
            project=project,
            config=config,
            name=name,
            notes=notes,
            tags=tags,
        )

    def log_metrics(self, metrics):
        """
        Log a dictionary of metrics.
        :param metrics: Dictionary of metrics (e.g., {"loss": 0.1, "accuracy": 0.9})
        """
        wandb.log(metrics)

    def log_model(self, model_path):
        """
        Save and log model checkpoints.
        :param model_path: Path to the saved model file.
        """
        wandb.save(model_path)

    def log_image(self, key, image, caption=None):
        """
        Log an image or visualization.
        :param key: The wandb key for this log entry.
        :param image: The image to log.
        :param caption: Optional caption for the image.
        """
        wandb.log({key: wandb.Image(image, caption=caption)})

    def log_3d_visualization(self, key, plot):
        """
        Log 3D visualizations (e.g., point clouds or bounding boxes).
        :param key: The wandb key for this log entry.
        :param plot: The plot object (e.g., Matplotlib, Plotly).
        """
        wandb.log({key: wandb.Html(plot.to_html())})  # Example for Plotly

    def finish(self):
        """
        Finish the wandb run.
        """
        self.run.finish()