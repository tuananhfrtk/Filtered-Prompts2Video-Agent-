# Filtered-Prompts2Video-Agent-
# AI Video Generator

This application leverages advanced AI models to generate videos from prompts, enhancing them for clarity and engagement. It integrates with Google Drive for video storage and provides a user-friendly interface using Streamlit.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [License](#license)

## Features
- Enhance prompts for video generation.
- Generate videos using AI models.
- Upload generated videos to Google Drive.
- User-friendly interface with Streamlit.

## Requirements
- Python 3.8 or higher
- Required packages listed in `requirements.txt`

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add the following variables:
   ```plaintext
   BASE_URL=https://api.groq.com/openai/v1
   GROQ_API_KEY=<your_groq_api_key>
   RUNWAY_API_KEY=<your_runway_api_key>
   GOOGLE_DRIVE_CREDS=<your_google_drive_credentials>
   OUTPUT_DIR=outputs
   DATASET_PATH=./prompts.csv
   ```

## Running the Application

To run the application, execute the following command:
```bash
python streamlit_ui.py
```

This will start the Streamlit server, and you can access the application in your web browser at `http://localhost:8501`.

## Usage

1. **Select Prompts:** Choose harmful prompts from the database to process.
2. **Enhance Prompts:** Use the auto-enhance feature or manually edit the prompts.
3. **Generate Videos:** Click the button to generate videos based on the enhanced prompts.
4. **View Results:** Watch the generated videos and download them from Google Drive.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
