# AI-Powered Data Exploratory Dat Analysis

This project demonstrates how to use an AI model (Claude from Anthropic) to interact with data stored in an Amazon RDS instance or a local file (CSV, Parquet, or JSON) and present the results in a Streamlit interface.

## Prerequisites

- Python 3.9
- Docker (optional)

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/eda-genai.git
cd eda-genai
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the project directory and add your RDS connection details (if using the database option) and Anthropic API key:
```
RDS_HOST=<your_rds_host>
RDS_DATABASE=<your_rds_database>
RDS_USER=<your_rds_user>
RDS_PASSWORD=<your_rds_password>
RDS_PORT=<your_rds_port>
ANTHROPIC_API_KEY=<your_anthropic_api_key>
```

## Usage

1. Run the Streamlit app:
```
streamlit run app.py
```
2. Open the app in your web browser.

3. Choose whether you want to explore a single file or data in a database.

4. If exploring a single file:
   - Enter your Anthropic API key in the input field.
   - Upload a CSV, Parquet, or JSON file containing your data.
   - Enter your query or prompt related to the uploaded data in the text area.
   - Click the "Submit" button to get the response from the AI model.

5. If exploring data in a database:
   - Enter your RDS connection details in the input fields.
   - Enter your Anthropic API key in the input field.
   - Enter your query or prompt related to the data in the database in the text area.
   - Click the "Submit" button to get the response from the AI model.

## Docker Deployment

You can also deploy the application using Docker:

1. Build the Docker image:
```
docker build -t eda-genai
```
2. Run the Docker container:
```
docker run -p 8501:8501 -e RDS_HOST=<your_rds_host> -e RDS_DATABASE=<your_rds_database> -e RDS_USER=<your_rds_user> -e RDS_PASSWORD=<your_rds_password> -e RDS_PORT=<your_rds_port> -e ANTHROPIC_API_KEY=<your_anthropic_api_key> ai-data-exploration
```
Replace the environment variables with your actual RDS connection details and Anthropic API key.

3. Open the app in your web browser at `http://localhost:8501`.

## License

This project is licensed under the [MIT License](LICENSE).
