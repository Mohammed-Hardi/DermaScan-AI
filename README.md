# DermaScan AI

DermaScan AI is a skin lesion classification project built with PyTorch and Streamlit. It uses the HAM10000 dataset to train a 5-class classifier for common skin lesion types.

## Features

- Streamlit web app for uploading dermoscopy images
- 5-class skin lesion prediction
- Includes model training and prediction pipeline
- Uses a clean project structure with `src/` modules

## Getting Started

### Prerequisites

- Python 3.10+ recommended
- Git installed

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in your terminal.

### Train the model

Place the HAM10000 dataset in `data/HAM10000` and confirm the `HAM10000_metadata.csv` file is available.

```bash
python src/train.py
```

Training may take 30-60 minutes on CPU depending on your machine.

## Project Structure

- `app.py` - Streamlit app entry point
- `src/` - model, dataset, prediction, training, and evaluation modules
- `config.py` - shared project settings and paths
- `requirements.txt` - Python dependencies
- `models/` - trained model checkpoints and status files
- `data/` - dataset files
- `outputs/` - temporary outputs and generated artifacts

## Notes

- The repository includes a `.gitignore` to skip virtual environments, data outputs, and model checkpoints.
- This project is for educational purposes only and not a medical diagnostic tool.

## License

Include license information here if desired.
