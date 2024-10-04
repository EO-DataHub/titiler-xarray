FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the pyproject dependencies
COPY pyproject.toml pdm.lock* /app/
RUN pip install --upgrade pip \
    && pip install pdm
  
# Installing directly from the pyproject wasn't working...
# So lock the dependencies, export to requirements.txt and install with pip
RUN pdm lock && pdm export -f requirements --prod --without-hashes --output requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# TODO: Could move to pyproject.toml, but then might get merge conflicts when syncing with upstream fork
RUN pip install uvicorn

COPY . /app
EXPOSE 8000
ENV TEST_ENVIRONMENT=true
CMD ["uvicorn", "titiler.xarray.main:app", "--host", "0.0.0.0", "--port", "8000"]