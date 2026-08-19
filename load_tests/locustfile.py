from pathlib import Path
from locust import HttpUser, constant, events, task

image_bytes = None
image_name = None

@events.init_command_line_parser.add_listener
def add_custom_arguments(parser):
    parser.add_argument(
        "--image",
        type=str,
        default="images/dog2.jpg",
        help="Image file to send to /predict"
    )

@events.test_start.add_listener
def load_image(environment, **kwargs):
    global image_bytes, image_name

    image_path = Path(environment.parsed_options.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image could not be found: {image_path}")

    image_bytes = image_path.read_bytes()
    image_name = image_path.name

    print(f"Load testing with image: {image_path}")

class InferenceUser(HttpUser):

    wait_time = constant(0)

    @task
    def predict(self):

        files = {
            "file": (image_name, image_bytes, "image/jpeg")
            }

        self.client.post("/predict", files=files, name="/predict")