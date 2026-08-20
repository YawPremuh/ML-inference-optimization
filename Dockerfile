FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install \
        --no-cache-dir \
        torch==2.13.0 \
        torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install \
        --no-cache-dir \
        -r requirements-api.txt \
    && python -m pip check \
    && python -m pip uninstall -y pip

RUN python -c \
    "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"

COPY app ./app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]