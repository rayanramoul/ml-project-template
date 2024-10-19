"""This is an example of a LitServe api for the Mnist LightningModule."""

import lightning
import litserve as ls
import torch
from torchvision.transforms import transforms


# Define the LitServe API for serving the MNIST model
class MNISTServeAPI(ls.LitAPI):
    """LitServe API for serving the MNIST model."""

    def __init__(self, model_class: lightning.pytorch.LightningModule, checkpoint_path: str):
        """Initialize the MNISTServeAPI.

        Args:
            model_class: The LightningModule class to serve.
            checkpoint_path: The path to the model checkpoint.
        """
        self.checkpoint_path = checkpoint_path
        self.model_class = model_class

    def setup(self, device: str):
        """Setup is called once at startup.

        Load the model, set the device, and prepare any other necessary components.
        """
        # Load the trained MNIST model (ensure model weights are loaded properly here)
        self.model = self.model_class.load_from_checkpoint(self.checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)  # Move the model to the appropriate device (CPU or GPU)
        self.model.eval()  # Set the model to evaluation mode

        # Define transforms that match the training data processing pipeline
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def decode_request(self, request: dict):
        """Decode the incoming request and prepare the input for the model."""
        # Convert the request payload into a tensor for model input
        image_data = request["image"]
        # Ensure that the image is a tensor of shape [1, 28, 28] (MNIST image dimensions)
        image_tensor = torch.tensor(image_data).unsqueeze(0)  # Add a batch dimension
        return self.transforms(image_tensor)  # Apply the necessary transformations

    def predict(self, x: torch.Tensor):
        """Run inference using the MNIST model and return the prediction."""
        # Forward pass through the model
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0))  # Add batch dimension for inference
            preds = torch.argmax(logits, dim=1)  # Get the predicted class
        return {"prediction": preds.item()}  # Return the prediction as a dictionary

    def encode_response(self, output: dict):
        """Encode the model's output into a response payload."""
        # Simply pass the output as the response
        return {"predicted_class": output["prediction"]}
