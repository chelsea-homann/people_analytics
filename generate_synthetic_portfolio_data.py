"""
=============================================================================
Synthetic Data Generator for People Analytics Portfolio
=============================================================================
Purpose:
    Generates synthetic test datasets for the full portfolio of people
    analytics scripts. All data is randomly generated and contains no
    real employee or organizational information.

Output Files:
    1. synthetic_workforce.csv       - Employee demographics & job data
    2. synthetic_career_changes.csv  - Internal job change history
    3. synthetic_survey_responses.csv - Likert-scale survey data
    4. synthetic_resumes.csv         - Resume text + structured features
    5. synthetic_social_posts.csv    - Social media post text

Usage:
    python generate_synthetic_portfolio_data.py
=============================================================================
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Shared reference lists ----
DEPARTMENTS = [
    "Finance", "Human Resources", "Marketing", "Sales", "Engineering",
    "Operations", "Legal", "Product", "Customer Service", "IT",
    "Research", "Supply Chain", "Communications", "Risk Management"
]

JOB_PROFILES = [
    "Analyst I", "Analyst II", "Senior Analyst", "Associate",
    "Senior Associate", "Specialist", "Senior Specialist",
    "Manager", "Senior Manager", "Director", "Senior Director",
    "VP", "Coordinator", "Administrator", "Consultant"
]

JOB_LEVELS = ["Entry", "Professional", "Senior Professional", "Manager",
              "Senior Manager", "Director", "VP"]

GRADES = list(range(5, 18))

LOCATIONS = ["Northeast", "Southeast", "Midwest", "West", "Remote"]

DEGREE_TYPES = ["None", "Associates", "Bachelors", "Masters", "Doctorate"]

CHANGE_REASONS = [
    "Promotion > New Job > Higher Level",
    "Lateral Move > New Job > Same Level",
    "Demotion > New Job > Lower Level"
]

TERM_REASONS = [
    "Voluntary - New Opportunity", "Voluntary - Relocation",
    "Voluntary - Career Change", "Voluntary - Retirement",
    "Involuntary - Performance", "Involuntary - Restructuring",
    "Voluntary - Return to School", "Voluntary - Personal"
]

TERM_CATEGORIES = ["Voluntary", "Involuntary", "Retirement"]

LEAVE_TYPES = ["None", "Medical", "Parental", "Family Care", "Personal", "Military"]

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
    "Avery", "Jamie", "Sage", "Drew", "Blake", "Hayden", "Parker",
    "Dakota", "Finley", "Rowan", "Emery", "Reese", "River",
    "Cameron", "Skyler", "Logan", "Sydney", "Kendall", "Peyton",
    "Marley", "Addison", "Charlie", "Elliot"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris",
    "Clark", "Lewis", "Robinson", "Walker", "Hall", "Allen", "Young",
    "King", "Wright", "Scott"
]


def random_date(start_year, end_year):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


# ==========================================================================
# 1. SYNTHETIC WORKFORCE (500 rows)
# ==========================================================================
def generate_workforce(n=500):
    print(f"Generating synthetic_workforce.csv ({n} rows)...")

    employee_ids = [f"E{str(i).zfill(5)}" for i in range(1, n + 1)]

    # Create manager hierarchy (first 50 are managers)
    manager_ids = employee_ids[:50]

    records = []
    for i, eid in enumerate(employee_ids):
        gender = np.random.choice(["Female", "Male"], p=[0.52, 0.48])
        minority = np.random.choice(["Minority", "Non-Minority"], p=[0.35, 0.65])
        is_manager = 1 if i < 50 else np.random.choice([0, 1], p=[0.85, 0.15])

        hire_date = random_date(2010, 2023)
        tenure_years = round((datetime(2024, 1, 1) - hire_date).days / 365.25, 1)

        # Some employees are terminated
        terminated = np.random.choice([0, 1], p=[0.75, 0.25])
        if terminated:
            term_date = hire_date + timedelta(days=random.randint(180, 2500))
            if term_date > datetime(2024, 12, 31):
                term_date = None
                terminated = 0
                term_reason = None
                term_category = None
            else:
                term_reason = random.choice(TERM_REASONS)
                term_category = "Voluntary" if "Voluntary" in term_reason else (
                    "Retirement" if "Retirement" in term_reason else "Involuntary")
        else:
            term_date = None
            term_reason = None
            term_category = None

        dept = random.choice(DEPARTMENTS)
        job_level_idx = min(max(0, int(np.random.normal(2, 1.5))), 6)
        grade = random.choice(GRADES[:10]) if job_level_idx < 3 else random.choice(GRADES[5:])

        # Salary correlated with grade and minority status (for pay equity analysis)
        base_salary = 35000 + grade * 6000 + np.random.normal(0, 5000)
        if minority == "Minority":
            base_salary *= np.random.uniform(0.95, 1.02)  # slight gap for analysis
        if is_manager:
            base_salary *= 1.15
        base_salary = round(max(30000, base_salary), 2)

        leave = random.choice(LEAVE_TYPES) if np.random.random() < 0.2 else "None"

        mgr_id = random.choice(manager_ids) if i >= 50 else (
            random.choice(manager_ids[:10]) if i >= 10 else None)

        records.append({
            "employee_id": eid,
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "department": dept,
            "job_profile": random.choice(JOB_PROFILES),
            "job_level": JOB_LEVELS[job_level_idx],
            "grade": grade,
            "annual_salary": base_salary,
            "gender": gender,
            "minority_status": minority,
            "veteran": np.random.choice(["Yes", "No"], p=[0.08, 0.92]),
            "disability": np.random.choice(["Yes", "No"], p=[0.05, 0.95]),
            "age": random.randint(22, 65),
            "age_group": None,  # filled below
            "hire_date": hire_date.strftime("%m/%d/%Y"),
            "term_date": term_date.strftime("%m/%d/%Y") if term_date else None,
            "term_reason": term_reason,
            "term_category": term_category,
            "manager_id": mgr_id,
            "location": random.choice(LOCATIONS),
            "is_manager": is_manager,
            "highest_degree": random.choice(DEGREE_TYPES),
            "tenure_years": tenure_years,
            "exempt": np.random.choice([0, 1], p=[0.3, 0.7]),
            "grade_profile": random.choice(["Standard", "Market Premium", "Technical"]),
            "time_in_job_years": round(random.uniform(0.5, tenure_years + 0.5), 1),
            "leave_type": leave,
            "leave_start": (random_date(2018, 2023).strftime("%m/%d/%Y")
                           if leave != "None" else None),
            "leave_end": None,  # filled below
            "employee_type": "Regular"
        })

    df = pd.DataFrame(records)

    # Fill age groups
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 100],
                              labels=["Under 30", "30-39", "40-49", "50-59", "60+"])

    # Fill leave_end for those with leave
    for idx in df[df["leave_type"] != "None"].index:
        start = datetime.strptime(df.at[idx, "leave_start"], "%m/%d/%Y")
        end = start + timedelta(days=random.randint(14, 180))
        df.at[idx, "leave_end"] = end.strftime("%m/%d/%Y")

    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_workforce.csv"), index=False)
    return df


# ==========================================================================
# 2. SYNTHETIC CAREER CHANGES (800 rows)
# ==========================================================================
def generate_career_changes(workforce_df, n=800):
    print(f"Generating synthetic_career_changes.csv ({n} rows)...")

    # Select a subset of employees who have job changes
    eligible = workforce_df.sample(min(200, len(workforce_df)))

    records = []
    for _, emp in eligible.iterrows():
        num_changes = random.randint(1, 6)
        hire_date = datetime.strptime(emp["hire_date"], "%m/%d/%Y")

        current_level_idx = JOB_LEVELS.index(emp["job_level"]) if emp["job_level"] in JOB_LEVELS else 2
        current_dept = emp["department"]
        current_profile = emp["job_profile"]
        current_grade = emp["grade"]
        current_pay = emp["annual_salary"]

        for j in range(num_changes):
            effective_date = hire_date + timedelta(days=random.randint(365, 365 * (j + 2)))
            if effective_date > datetime(2024, 12, 31):
                break

            reason = np.random.choice(CHANGE_REASONS, p=[0.6, 0.3, 0.1])

            if "Higher" in reason:
                new_level_idx = min(current_level_idx + 1, 6)
                new_grade = min(current_grade + random.randint(1, 2), 17)
                new_pay = round(current_pay * random.uniform(1.05, 1.20), 2)
            elif "Lower" in reason:
                new_level_idx = max(current_level_idx - 1, 0)
                new_grade = max(current_grade - 1, 5)
                new_pay = round(current_pay * random.uniform(0.85, 0.98), 2)
            else:
                new_level_idx = current_level_idx
                new_grade = current_grade
                new_pay = round(current_pay * random.uniform(0.98, 1.05), 2)

            new_dept = random.choice(DEPARTMENTS) if random.random() < 0.3 else current_dept
            new_profile = random.choice(JOB_PROFILES)

            is_mgr_from = 1 if current_level_idx >= 3 else 0
            is_mgr_to = 1 if new_level_idx >= 3 else 0

            records.append({
                "employee_id": emp["employee_id"],
                "worker_name": emp["name"],
                "hire_date": emp["hire_date"],
                "effective_date": effective_date.strftime("%m/%d/%Y"),
                "reason": reason,
                "job_profile_from": current_profile,
                "job_profile_to": new_profile,
                "job_level_from": JOB_LEVELS[current_level_idx],
                "job_level_to": JOB_LEVELS[new_level_idx],
                "base_pay_from": current_pay,
                "base_pay_to": new_pay,
                "department_from": current_dept,
                "department_to": new_dept,
                "grade_from": current_grade,
                "grade_to": new_grade,
                "is_manager_from": is_mgr_from,
                "is_manager_to": is_mgr_to,
                "gender": emp["gender"],
                "minority_status": emp["minority_status"],
                "age": emp["age"],
                "days_in_job_before_change": random.randint(180, 1800),
                "days_in_position_before_change": random.randint(90, 1500),
                "active": np.random.choice(["Yes", "No"], p=[0.8, 0.2]),
                "diverse": emp["minority_status"]
            })

            # Update current state
            current_level_idx = new_level_idx
            current_dept = new_dept
            current_profile = new_profile
            current_grade = new_grade
            current_pay = new_pay

    df = pd.DataFrame(records[:n])
    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_career_changes.csv"), index=False)
    return df


# ==========================================================================
# 3. SYNTHETIC SURVEY RESPONSES (500 rows)
# ==========================================================================
def generate_survey_responses(n=500):
    print(f"Generating synthetic_survey_responses.csv ({n} rows)...")

    records = []

    # Define survey item groups
    engagement_items = ["Q1_1", "Q1_2", "Q1_3", "Q1_4"]
    manager_items = ["Q2_1", "Q2_2", "Q2_3", "Q2_4"]
    inclusion_items = ["Q3_1", "Q3_2", "Q3_3", "Q3_4", "Q3_5", "Q3_6"]
    org_items = ["Q4_1", "Q4_2", "Q4_3", "Q4_4"]

    # Wellbeing items (for correlation analysis)
    wellbeing_items = [f"WB_{i}" for i in range(1, 21)]

    all_items = engagement_items + manager_items + inclusion_items + org_items + wellbeing_items

    for i in range(n):
        # Create correlated responses within constructs
        eng_base = np.clip(np.random.normal(3.8, 0.8), 1, 5)
        mgr_base = np.clip(np.random.normal(3.6, 0.9), 1, 5)
        inc_base = np.clip(np.random.normal(3.7, 0.85), 1, 5)
        org_base = np.clip(np.random.normal(3.5, 0.9), 1, 5)
        wb_base = np.clip(np.random.normal(3.4, 0.95), 1, 5)

        row = {"respondent_id": f"R{str(i+1).zfill(4)}"}

        for item in engagement_items:
            row[item] = int(np.clip(round(eng_base + np.random.normal(0, 0.5)), 1, 5))
        for item in manager_items:
            row[item] = int(np.clip(round(mgr_base + np.random.normal(0, 0.5)), 1, 5))
        for item in inclusion_items:
            row[item] = int(np.clip(round(inc_base + np.random.normal(0, 0.5)), 1, 5))
        for item in org_items:
            row[item] = int(np.clip(round(org_base + np.random.normal(0, 0.5)), 1, 5))
        for item in wellbeing_items:
            row[item] = int(np.clip(round(wb_base + np.random.normal(0, 0.6)), 1, 5))

        row["department"] = random.choice(DEPARTMENTS)
        row["gender"] = np.random.choice(["Female", "Male"], p=[0.52, 0.48])
        row["management_level"] = np.random.choice(
            ["Individual Contributor", "Manager", "Senior Leader"], p=[0.75, 0.20, 0.05])
        row["ethnicity"] = np.random.choice(["Minority", "Non-Minority"], p=[0.35, 0.65])
        row["job_level"] = random.choice(JOB_LEVELS)
        row["is_manager"] = 1 if row["management_level"] != "Individual Contributor" else 0
        row["campus"] = random.choice(LOCATIONS)
        row["performance_rating"] = np.random.choice(
            ["Exceeds", "Meets", "Below"], p=[0.25, 0.65, 0.10])
        row["manager_id"] = f"M{str(random.randint(1, 50)).zfill(4)}"
        row["manager_name"] = f"Manager {random.randint(1, 50)}"

        # Terminated (for turnover prediction) — correlated with low engagement
        avg_eng = np.mean([row[q] for q in engagement_items])
        term_prob = 0.05 if avg_eng > 3.5 else 0.25 if avg_eng > 2.5 else 0.50
        row["terminated"] = np.random.choice([0, 1], p=[1 - term_prob, term_prob])

        # ERG participation (for I&D analysis)
        row["erg_participations"] = np.random.choice([0, 1, 2, 3], p=[0.5, 0.25, 0.15, 0.10])

        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_survey_responses.csv"), index=False)
    return df


# ==========================================================================
# 4. SYNTHETIC RESUMES (300 rows)
# ==========================================================================
def generate_resumes(n=300):
    print(f"Generating synthetic_resumes.csv ({n} rows)...")

    SKILLS = [
        "project management", "data analysis", "team leadership",
        "customer service", "communication", "problem solving",
        "Microsoft Office", "Python", "SQL", "financial analysis",
        "strategic planning", "process improvement", "agile methodology",
        "stakeholder management", "budgeting", "risk assessment",
        "quality assurance", "training development", "regulatory compliance",
        "business development"
    ]

    EDUCATION = [
        "Bachelor of Science in Business Administration",
        "Master of Business Administration",
        "Bachelor of Arts in Psychology",
        "Master of Science in Data Analytics",
        "Bachelor of Science in Finance",
        "Bachelor of Arts in Communications",
        "Master of Science in Human Resources",
        "Bachelor of Science in Computer Science",
        "Associate Degree in Business",
        "Bachelor of Science in Accounting"
    ]

    EXPERIENCE_TEMPLATES = [
        "Managed a team of {n} employees in {dept} department. Responsible for {skill1} and {skill2}.",
        "Led {skill1} initiatives resulting in improved operational efficiency across {dept}.",
        "Developed and implemented {skill1} strategies for the {dept} division.",
        "Collaborated with cross-functional teams on {skill1} and {skill2} projects.",
        "Spearheaded {skill1} programs that enhanced {skill2} outcomes.",
        "Oversaw {skill1} operations and drove {skill2} improvements.",
        "Designed and executed {skill1} frameworks supporting {dept} objectives."
    ]

    records = []
    for i in range(n):
        num_skills = random.randint(3, 8)
        selected_skills = random.sample(SKILLS, num_skills)
        education = random.choice(EDUCATION)
        years_exp = random.randint(1, 25)

        # Build resume text
        resume_parts = [f"EDUCATION: {education}."]
        resume_parts.append(f"EXPERIENCE: {years_exp} years of professional experience.")

        for _ in range(random.randint(2, 4)):
            template = random.choice(EXPERIENCE_TEMPLATES)
            text = template.format(
                n=random.randint(3, 20),
                dept=random.choice(DEPARTMENTS),
                skill1=random.choice(selected_skills),
                skill2=random.choice(selected_skills)
            )
            resume_parts.append(text)

        resume_parts.append(f"SKILLS: {', '.join(selected_skills)}.")
        resume_text = " ".join(resume_parts)

        # Structured features
        gpa = round(random.uniform(2.5, 4.0), 2)
        internal = np.random.choice([0, 1], p=[0.7, 0.3])
        referral = np.random.choice([0, 1], p=[0.6, 0.4])

        # Outcome: 90-day voluntary turnover (class imbalanced ~10%)
        turnover_prob = 0.05 if years_exp > 5 else 0.15
        if "team leadership" in selected_skills:
            turnover_prob *= 0.7
        outcome = np.random.choice([0, 1], p=[1 - turnover_prob, turnover_prob])

        records.append({
            "candidate_id": f"C{str(i+1).zfill(4)}",
            "resume_text": resume_text,
            "years_experience": years_exp,
            "education_level": education.split(" in ")[0] if " in " in education else education,
            "gpa": gpa,
            "internal_candidate": internal,
            "referral": referral,
            "num_skills_listed": num_skills,
            "department_applied": random.choice(DEPARTMENTS),
            "source": np.random.choice(
                ["Job Board", "Referral", "Career Site", "Campus", "Agency"],
                p=[0.3, 0.25, 0.25, 0.1, 0.1]),
            "outcome_90day_vol": outcome
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_resumes.csv"), index=False)
    return df


# ==========================================================================
# 5. SYNTHETIC SOCIAL POSTS (500 rows)
# ==========================================================================
def generate_social_posts(n=500):
    print(f"Generating synthetic_social_posts.csv ({n} rows)...")

    TOPICS = ["workplace", "technology", "leadership", "innovation"]

    POSITIVE_PHRASES = [
        "Great progress on our new initiative",
        "Excited about the future of work",
        "Innovation drives success",
        "Proud of what the team accomplished",
        "Amazing results from the latest project",
        "Thrilled to see leadership investing in people",
        "Technology is transforming how we collaborate",
        "Outstanding teamwork and dedication",
        "Love the new direction we are heading",
        "Best practices in employee engagement"
    ]

    NEUTRAL_PHRASES = [
        "Interesting article about industry trends",
        "New report on workforce analytics",
        "Conference session on digital transformation",
        "Reading about changes in the market",
        "Webinar on talent management strategies",
        "Survey results are in for this quarter",
        "Reviewing the latest benchmarking data",
        "Panel discussion on the future of HR tech"
    ]

    NEGATIVE_PHRASES = [
        "Disappointing lack of transparency",
        "Struggling with outdated tools and processes",
        "Frustrated with the slow pace of change",
        "Leadership needs to listen more to employees",
        "Poor communication about organizational changes",
        "Burnout is a real problem that needs addressing",
        "Too much bureaucracy stifling innovation",
        "Need better work-life balance support"
    ]

    records = []
    for i in range(n):
        topic = random.choice(TOPICS)
        sentiment_type = np.random.choice(
            ["positive", "neutral", "negative"], p=[0.45, 0.30, 0.25])

        if sentiment_type == "positive":
            text = random.choice(POSITIVE_PHRASES)
        elif sentiment_type == "neutral":
            text = random.choice(NEUTRAL_PHRASES)
        else:
            text = random.choice(NEGATIVE_PHRASES)

        # Add hashtag and some variation
        text += f" #{topic} #{random.choice(['analytics', 'peoplefirst', 'futureofwork', 'hrtech', 'culture'])}"

        records.append({
            "post_id": f"P{str(i+1).zfill(5)}",
            "text": text,
            "topic": topic,
            "hashtag": f"#{topic}",
            "timestamp": random_date(2022, 2024).strftime("%Y-%m-%d %H:%M:%S"),
            "platform": np.random.choice(
                ["Twitter", "LinkedIn", "BlueSky"], p=[0.4, 0.45, 0.15]),
            "likes": random.randint(0, 500),
            "shares": random.randint(0, 100)
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_social_posts.csv"), index=False)
    return df


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Synthetic Portfolio Datasets")
    print("=" * 60)

    workforce = generate_workforce(500)
    career = generate_career_changes(workforce, 800)
    survey = generate_survey_responses(500)
    resumes = generate_resumes(300)
    social = generate_social_posts(500)

    print("\n" + "=" * 60)
    print("All datasets generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    print(f"  synthetic_workforce.csv:        {len(workforce)} rows")
    print(f"  synthetic_career_changes.csv:   {len(career)} rows")
    print(f"  synthetic_survey_responses.csv: {len(survey)} rows")
    print(f"  synthetic_resumes.csv:          {len(resumes)} rows")
    print(f"  synthetic_social_posts.csv:     {len(social)} rows")
