# An agent with LangChain for query and visual data

Use the package manager [pip](https://pip.pypa.io/en/stable/%29) to install the required packages.

## Setup

Follow these steps to set up the project:

### 1. Create Virtual Environment

```bash
# On Windows
python -m venv venv

# On macOS and Linux
python3 -m venv venv
```

### 2. Activate Virtual Environment

```bash
# On Windows
.\venv\Scripts\activate

# On macOS and Linux
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Edit the `.env` file and add necessary environment variables.

## Usage

To run the app, use the following command:

```bash
streamlit run app.py
```

