# Smart SGPA/CGPA Calculator - Complete Build Guide

> **A step-by-step blueprint from zero to a full-stack SGPA calculator with AI integration**

---

## 📋 Table of Contents

1. [Project Overview](#overview)
2. [Phase 1: Backend Setup](#phase-1-backend-setup)
   - [Step 1: Create Folders](#step-1-create-folders)
   - [Step 2: Python Environment](#step-2-set-up-python-environment)
   - [Step 3: Install Libraries](#step-3-install-python-libraries)
   - [Step 4: Get API Key](#step-4-get-your-secret-api-key)
   - [Step 5: Create Backend Files](#step-5-create-backend-files)
3. [Phase 2: Frontend Setup](#phase-2-frontend-setup)
   - [Step 1: Create React Project](#step-1-create-the-react-project)
   - [Step 2: Install JS Libraries](#step-2-install-javascript-libraries)
   - [Step 3: Create Folders](#step-3-create-frontend-folders)
   - [Step 4: Create Frontend Files](#step-4-create-frontend-files)
4. [Phase 3: Run the Project](#phase-3-the-startup-file)
5. [Project Structure](#project-structure)

---

## Overview

This document contains:
- **Complete step-by-step instructions** to build from scratch
- **All final, correct code** for every single file in one place
- **Organized with clear sections** — guide text separated from code boxes
- **Professional formatting** with syntax highlighting and easy navigation

Everything is organized so you can follow it linearly, file by file, and have a working project at the end.

---

## Phase 1: Backend Setup

### Step 1: Create Folders

Your project folder structure should look like this:

```
sgpa-project/
├── backend/
│   ├── venv/
│   ├── .env
│   ├── .gitignore
│   ├── parser.py
│   └── app.py
└── frontend/
```

**Instructions:**
1. Create one main project folder called `sgpa-project`
2. Inside `sgpa-project`, create a subfolder called `backend`

---

### Step 2: Set Up Python Environment

The Python virtual environment (venv) is like a "sandbox" where we install only the packages we need for this project.

**Instructions:**

Open your terminal and navigate to your backend folder:

```bash
cd path/to/sgpa-project/backend
```

Create the virtual environment:

```bash
python -m venv venv
```

Activate it (on Windows):

```bash
.\venv\Scripts\activate
```

After activation, your terminal prompt should start with `(venv)`. This means you're now "inside" the isolated Python environment.

---

### Step 3: Install Python Libraries

With your `(venv)` active, install all our dependencies in one command:

```bash
pip install flask flask-cors pymupdf pandas scikit-learn requests python-dotenv
```

**What each library does:**
- **Flask**: Web server framework
- **flask-cors**: Allows React (frontend) to talk to Flask (backend)
- **pymupdf**: Reads and parses PDF files
- **pandas**: Data manipulation (optional, for future use)
- **scikit-learn**: Machine learning for predictions
- **requests**: Makes HTTP calls to Gemini AI
- **python-dotenv**: Loads secret API keys from `.env` file

---

### Step 4: Get Your Secret API Key

The Gemini API is free and provides AI analysis features.

**Instructions:**
1. Go to [https://aistudio.google.com/app](https://aistudio.google.com/app)
2. Click "Get API Key" and create a new free key
3. Copy the key (looks like `AIzaSy...`)
4. Save it somewhere temporarily — you'll paste it in the next section

---

### Step 5: Create Backend Files

You will create **4 text files** inside your `backend/` folder.

#### File 1: `.env` (Your Secret Key)

**Purpose:** Stores your Gemini API key securely. This file is *never* pushed to GitHub.

**Location:** `backend/.env`

**Content:**

```
GEMINI_API_KEY=YOUR_KEY_HERE
```

Replace `YOUR_KEY_HERE` with the Gemini API key you just copied.

---

#### File 2: `.gitignore` (Protect Your Secrets)

**Purpose:** Tells Git to ignore files with sensitive data (like `.env`) and temporary files.

**Location:** `backend/.gitignore`

**Content:**

```
# Ignore the secret API key file
.env

# Ignore the Python virtual environment folder
venv/

# Ignore Python cache files
__pycache__/
*.pyc

# Ignore temp files
*.tmp
temp.pdf
```

---

#### File 3: `parser.py` (The PDF Parser & Calculator Engine)

**Purpose:** Extracts marks from PDF, calculates SGPA, handles all the math logic.

**Location:** `backend/parser.py`

**Content:**

```python
import fitz  # PyMuPDF
import re    # Regular Expressions
import json
import sys   # We'll use this to exit if tests fail

# --- 1. THE LOGIC YOU PROVIDED ---

# This is the "Credits Map" based on the OFFICIAL VTU SCHEME.
CREDITS_MAP = {
    'BCS401': 3,
    'BCS402': 4,
    'BCS403': 4,
    'BCSL404': 1,
    'BBOC407': 2,
    'BUHK408': 1,
    'BPEK459': 0,  # Non-credit
    'BCS405A': 3,
    'BDSL456B': 1,
    'BCSL405': 1, # Added from your examples
    'BCSL406': 1, # Added from your examples
}

# This function converts Total Marks -> Grade Points, based on your logic.
def get_grade_points(total_marks, result_status):
    """
    Converts total marks and pass/fail status into VTU grade points.
    """
    if result_status == 'F':
        return 0

    if 90 <= total_marks <= 100:
        return 10
    elif 80 <= total_marks <= 89:
        return 9
    elif 70 <= total_marks <= 79:
        return 8
    elif 60 <= total_marks <= 69:
        return 7
    elif 55 <= total_marks <= 59:
        return 6
    elif 50 <= total_marks <= 54:
        return 5
    elif 40 <= total_marks <= 49:
        return 4
    else:
        return 0

# --- 2. SELF-TESTING UNIT TEST ---
def test_grade_logic():
    """
    A simple unit test to verify the get_grade_points function.
    """
    print("Running grade logic unit tests...")
    try:
        assert get_grade_points(100, 'P') == 10
        assert get_grade_points(90, 'P') == 10
        assert get_grade_points(89, 'P') == 9
        assert get_grade_points(80, 'P') == 9
        assert get_grade_points(79, 'P') == 8
        assert get_grade_points(70, 'P') == 8
        assert get_grade_points(69, 'P') == 7
        assert get_grade_points(60, 'P') == 7
        assert get_grade_points(59, 'P') == 6
        assert get_grade_points(55, 'P') == 6
        assert get_grade_points(54, 'P') == 5
        assert get_grade_points(50, 'P') == 5
        assert get_grade_points(49, 'P') == 4
        assert get_grade_points(40, 'P') == 4
        assert get_grade_points(39, 'P') == 0
        assert get_grade_points(55, 'F') == 0 # Should be 0, not 6
        assert get_grade_points(95, 'F') == 0 # Should be 0, not 10
        print("✓ All grade logic tests passed!")
    except AssertionError:
        print("✗ CRITICAL: Grade logic test failed!")
        sys.exit(1) # Stop the script if logic is broken

# --- 3. THE PARSER FUNCTION ---
def parse_marks_card(pdf_file_path):
    """
    Parses a digital marks card PDF, extracts key information,
    and calculates SGPA using VTU 2022 scheme logic.
    """

    full_text = ""
    try:
        # --- A. EXTRACT ALL TEXT ---
        doc = fitz.open(pdf_file_path)
        for page in doc:
            full_text += page.get_text()
        doc.close()

        print(f"--- Extracted {len(full_text)} characters from PDF.")

        # --- B. FIND PATTERNS WITH ROBUST REGEX ---
        usn_pattern = re.compile(r"University Seat Number\s*:\s*(\w+)")
        name_pattern = re.compile(r"Student Name\s*:\s*(.+)")

        subject_pattern = re.compile(
            r"([A-Z]{3,}\d{3}[A-Z]?)\s*"  # 1: Subject Code
            r"(.+?)\s*"                  # 2: Subject Name
            r"(\d+)\s*"                  # 3: Internal Marks
            r"(\d+)\s*"                  # 4: External Marks
            r"(\d+)\s*"                  # 5: Total Marks
            r"([PF])\s*",                # 6: Result (P or F)
            re.DOTALL
        )

        usn = usn_pattern.search(full_text).group(1).strip()
        name = name_pattern.search(full_text).group(1).strip()

        subjects = []
        total_credits_attempted = 0
        total_grade_points_earned = 0

        subject_matches = subject_pattern.findall(full_text)
        print(f"--- Found {len(subject_matches)} subjects.")

        for match in subject_matches:
            # Use match[0], match[1] etc. since findall() returns tuples
            code = match[0].strip()
            title = match[1].strip().replace("\n", " ")
            internal = int(match[2])
            external = int(match[3])
            total = int(match[4])
            result = match[5].strip()

            credits = CREDITS_MAP.get(code, 0)

            if code not in CREDITS_MAP:
                print(f"⚠️  Warning: Subject code '{code}' not found in CREDITS_MAP. Using 0 credits.")

            points = get_grade_points(total, result)

            subjects.append({
                "code": code,
                "title": title,
                "internal": internal,
                "external": external,
                "total": total,
                "result": result,
                "credits": credits,
                "points": points
            })

            # Per VTU logic, all subjects (even failed) count to the total credits.
            total_credits_attempted += credits
            total_grade_points_earned += (points * credits)

        # --- E. CALCULATE SGPA ---
        sgpa = 0.0
        if total_credits_attempted > 0:
            sgpa = round(total_grade_points_earned / total_credits_attempted, 2)

        # --- F. CALCULATE PERCENTAGE ---
        # This is the *Semester* Percentage.
        percentage = round(sgpa * 10, 2)
        if percentage < 0:
            percentage = 0 

        # --- G. RETURN THE CLEAN DATA ---
        return {
            "status": "success",
            "usn": usn,
            "name": name,
            "sgpa": sgpa,
            "percentage": percentage,
            "total_credits_attempted": total_credits_attempted,
            "total_grade_points_earned": total_grade_points_earned,
            "subjects": subjects
        }

    # --- SPECIFIC ERROR HANDLING ---
    except AttributeError as e:
        return {
            "status": "error",
            "message": f"A required pattern was not found (e.g., USN or Name). Error: {e}",
            "raw_text": full_text
        }
    except ValueError as e:
        return {
            "status": "error",
            "message": f"Failed to convert marks to a number. Error: {e}",
            "raw_text": full_text
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
            "raw_text": full_text
        }

# --- This line runs our unit test ---
if __name__ == "__main__":
    test_grade_logic()
    print("\n parser.py is ready to be imported.")
```

---

#### File 4: `app.py` (The Flask Web Server)

**Purpose:** Creates the web API that the React app talks to. Receives PDFs, runs parser, calls Gemini AI.

**Location:** `backend/app.py`

**Content:**

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from parser import parse_marks_card # Our parser
from werkzeug.utils import secure_filename
import tempfile
import os
import numpy as np # Import numpy for our prediction
from sklearn.linear_model import LinearRegression # Import the model
import requests # For calling Gemini
import json     # For building Gemini payload
from dotenv import load_dotenv # For loading secret API key

# --- Load our secret .env file ---
load_dotenv() 

app = Flask(__name__)
# Allow our React app to talk to this server
CORS(app, origins=["http://localhost:3000"]) 
# Set a max file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Checks if the uploaded file is a PDF"""
    return '.' in filename and \\
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_cgpa_data(past_sgpas_str, new_sgpa):
    """
    Calculates CGPA, Percentage, Classification, and Prediction
    using the VTU 2022 "Simple Average" logic.
    """
    try:
        # --- 1. Prepare Data ---
        past_sgpas = [float(s.strip()) for s in past_sgpas_str.split(',') if s.strip()]
        all_sgpas = past_sgpas + [new_sgpa] # e.g., [8.0, 7.5, 8.2, 7.0]

        # --- 2. Calculate Current CGPA (Simple Average) ---
        cgpa = 0
        if all_sgpas:
            cgpa = round(sum(all_sgpas) / len(all_sgpas), 2)

        # --- 3. Calculate Percentage & Classification (VTU 2022 Logic) ---
        percentage = round(cgpa * 10, 2)

        classification = "Second Class (SC)"
        if cgpa >= 7.75:
            classification = "First Class with Distinction (FCD)"
        elif cgpa >= 6.75:
            classification = "First Class (FC)"

        # --- 4. Calculate Prediction ---
        prediction = None
        if len(all_sgpas) >= 2: # Need at least 2 points for a trend
            X = np.array(range(1, len(all_sgpas) + 1)).reshape(-1, 1)
            y = np.array(all_sgpas)

            model = LinearRegression()
            model.fit(X, y)

            next_sem_num = len(all_sgpas) + 1
            predicted_sgpa = model.predict(np.array([[next_sem_num]]))[0]
            predicted_sgpa = round(np.clip(predicted_sgpa, 0, 10), 2) # Clamp between 0-10

            new_sgpa_list = all_sgpas + [predicted_sgpa]
            predicted_cgpa = round(sum(new_sgpa_list) / len(new_sgpa_list), 2)

            prediction = {
                "past_trend": all_sgpas,
                "predicted_sgpa": predicted_sgpa,
                "predicted_cgpa": predicted_cgpa
            }

        # --- 5. Return all data ---
        return {
            "cgpa": cgpa,
            "percentage": percentage,
            "classification": classification,
            "prediction": prediction
        }

    except Exception as e:
        print(f"CGPA/Prediction error: {e}")
        return None # Fail silently

@app.route("/")
def index():
    """A simple test route to see if the server is running."""
    return jsonify({"status": "Flask API is running!"})

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    This is the main endpoint. It receives the PDF and past SGPAs,
    runs the parser, and then runs the CGPA calculation.
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    past_sgpas_str = request.form.get('past_sgpas', '')

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid or missing PDF file"}), 400

    temp_pdf_path = ""
    try:
        # Save the file to a secure, unique temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_pdf_path = temp_file.name

        # --- 1. Run our Parser Engine ---
        # This gives us the new SGPA
        results = parse_marks_card(temp_pdf_path)

        if results["status"] == "error":
            return jsonify(results), 500

        # --- 2. Run our new CGPA/Prediction Engine ---
        # We only run this if the user provided past SGPAs
        if past_sgpas_str:
            cgpa_data = calculate_cgpa_data(
                past_sgpas_str, 
                results["sgpa"]
            )
            # Add this new data (CGPA, Percentage, etc.) to our main results
            results.update(cgpa_data)

        return jsonify(results)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # Always delete the temporary file for privacy
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@app.route("/get-ai-tip", methods=["POST"])
def get_ai_tip():
    """
    Receives subject data from the frontend, sends it to Gemini,
    and returns an AI-generated study tip.
    """

    # 1. Get the data from the React app
    data = request.get_json()
    subjects = data.get('subjects')
    sgpa = data.get('sgpa')

    if not subjects:
        return jsonify({"status": "error", "message": "No subject data provided"}), 400

    # 2. Build a prompt for Gemini
    try:
        passed_subjects = [s for s in subjects if s['result'] == 'P' and s['credits'] > 0]
        worst_subject = min(passed_subjects, key=lambda x: x['points'])
        failed_subjects = [s for s in subjects if s['result'] == 'F']

        prompt = (f"A VTU data science student just got their 4th sem results. "
                  f"Their SGPA was {sgpa}. ")
        if failed_subjects:
            prompt += (f"They failed: {', '.join([s['title'] for s in failed_subjects])}. "
                       f"Their worst *passed* subject was {worst_subject['title']} (Grade Points: {worst_subject['points']}). ")
            prompt += "Please provide one paragraph of concise, actionable study advice focusing on how to recover from the failed subjects and improve."
        else:
            prompt += (f"Their worst passed subject was {worst_subject['title']} (Grade Points: {worst_subject['points']}). "
                       f"Please provide one paragraph of concise, actionable study advice on how to improve this specific subject.")
    except Exception:
        prompt = f"A VTU data science student just got an SGPA of {sgpa}. Provide one paragraph of concise, positive study advice."

    # 3. Call the Gemini API
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"status": "error", "message": "API key not configured."}), 500

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

        system_instruction = "You are a helpful and encouraging academic tutor for a data science engineering student."
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]}
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
        response.raise_for_status()

        api_result = response.json()
        tip = api_result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({"status": "success", "tip": tip})

    except requests.exceptions.Timeout:
        return jsonify({"status": "error", "message": "AI server (Gemini) timed out. Please try again."}), 504
    except requests.exceptions.RequestException as e:
        print(f"Gemini API Error: {e}")
        return jsonify({"status": "error", "message": f"AI server error: {e}"}), 500
    except (KeyError, IndexError):
        return jsonify({"status": "error", "message": "Failed to parse AI response."}), 500

if __name__ == "__main__":
    app.run(debug=True)
```

**Congratulations!** Backend is complete. Your Flask server is ready.

---

## Phase 2: Frontend Setup

### Step 1: Create the React Project

Open a **new terminal** (keep the backend terminal open too!).

Navigate to your root **sgpa-project** folder (the one *above* backend):

```bash
cd path/to/sgpa-project
npx create-react-app frontend
```

This creates a complete React project. It will take 1-2 minutes.

---

### Step 2: Install JavaScript Libraries

Navigate into the frontend folder:

```bash
cd frontend
npm install chart.js react-chartjs-2 prop-types
```

**What each library does:**
- **chart.js**: Core charting library
- **react-chartjs-2**: React wrapper for chart.js
- **prop-types**: Adds type checking to React components

---

### Step 3: Create Frontend Folders

Inside your `frontend/src` folder, create two new folders:

```
frontend/src/
├── components/      (← create this folder)
├── pages/          (← create this folder)
├── App.js
├── App.css
└── index.js
```

---

### Step 4: Create Frontend Files

You will create **7 files** inside your `frontend/src` folder.

#### File 1: `App.js`

**Pu
