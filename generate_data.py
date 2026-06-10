import os
import numpy as np
import pandas as pd

np.random.seed(42)

os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------
# 1. Concepts dataset
# ---------------------------------------------------------

concepts = pd.DataFrame([
    {
        "concept_id": "C01",
        "concept_name": "Probability Basics",
        "skill_area": "Probability",
        "difficulty": 1,
        "estimated_time_min": 25,
        "base_mastery_gain": 0.18,
        "content_type": "Video",
        "engagement_score": 0.86,
        "exam_importance": 5
    },
    {
        "concept_id": "C02",
        "concept_name": "Conditional Probability",
        "skill_area": "Probability",
        "difficulty": 2,
        "estimated_time_min": 30,
        "base_mastery_gain": 0.17,
        "content_type": "Practice",
        "engagement_score": 0.82,
        "exam_importance": 5
    },
    {
        "concept_id": "C03",
        "concept_name": "Bayes Rule",
        "skill_area": "Bayesian Reasoning",
        "difficulty": 2,
        "estimated_time_min": 35,
        "base_mastery_gain": 0.20,
        "content_type": "Interactive",
        "engagement_score": 0.88,
        "exam_importance": 5
    },
    {
        "concept_id": "C04",
        "concept_name": "Utility Theory",
        "skill_area": "Decision Theory",
        "difficulty": 2,
        "estimated_time_min": 30,
        "base_mastery_gain": 0.18,
        "content_type": "Reading",
        "engagement_score": 0.76,
        "exam_importance": 4
    },
    {
        "concept_id": "C05",
        "concept_name": "Expected Utility",
        "skill_area": "Decision Theory",
        "difficulty": 3,
        "estimated_time_min": 35,
        "base_mastery_gain": 0.19,
        "content_type": "Practice",
        "engagement_score": 0.80,
        "exam_importance": 5
    },
    {
        "concept_id": "C06",
        "concept_name": "Sequential Decision Making",
        "skill_area": "Sequential Decisions",
        "difficulty": 3,
        "estimated_time_min": 40,
        "base_mastery_gain": 0.20,
        "content_type": "Video",
        "engagement_score": 0.84,
        "exam_importance": 5
    },
    {
        "concept_id": "C07",
        "concept_name": "Markov Decision Processes",
        "skill_area": "Reinforcement Learning",
        "difficulty": 4,
        "estimated_time_min": 45,
        "base_mastery_gain": 0.22,
        "content_type": "Interactive",
        "engagement_score": 0.90,
        "exam_importance": 5
    },
    {
        "concept_id": "C08",
        "concept_name": "Value Iteration",
        "skill_area": "Reinforcement Learning",
        "difficulty": 4,
        "estimated_time_min": 45,
        "base_mastery_gain": 0.21,
        "content_type": "Practice",
        "engagement_score": 0.83,
        "exam_importance": 4
    },
    {
        "concept_id": "C09",
        "concept_name": "Q-learning",
        "skill_area": "Reinforcement Learning",
        "difficulty": 5,
        "estimated_time_min": 50,
        "base_mastery_gain": 0.24,
        "content_type": "Coding",
        "engagement_score": 0.91,
        "exam_importance": 5
    },
    {
        "concept_id": "C10",
        "concept_name": "Exploration vs Exploitation",
        "skill_area": "Reinforcement Learning",
        "difficulty": 4,
        "estimated_time_min": 35,
        "base_mastery_gain": 0.20,
        "content_type": "Interactive",
        "engagement_score": 0.89,
        "exam_importance": 5
    },
    {
        "concept_id": "C11",
        "concept_name": "Multi-Criteria Decision Making",
        "skill_area": "MCDM",
        "difficulty": 3,
        "estimated_time_min": 35,
        "base_mastery_gain": 0.18,
        "content_type": "Practice",
        "engagement_score": 0.81,
        "exam_importance": 4
    },
    {
        "concept_id": "C12",
        "concept_name": "Fuzzy Decision Making",
        "skill_area": "Fuzzy Logic",
        "difficulty": 4,
        "estimated_time_min": 40,
        "base_mastery_gain": 0.19,
        "content_type": "Interactive",
        "engagement_score": 0.85,
        "exam_importance": 4
    }
])

concepts.to_csv("data/concepts.csv", index=False)

# ---------------------------------------------------------
# 2. Concept dependency graph
# source = prerequisite
# target = next concept
# ---------------------------------------------------------

prerequisites = pd.DataFrame([
    {"source": "C01", "target": "C02", "dependency_strength": 0.90},
    {"source": "C02", "target": "C03", "dependency_strength": 0.95},
    {"source": "C01", "target": "C04", "dependency_strength": 0.70},
    {"source": "C04", "target": "C05", "dependency_strength": 0.90},
    {"source": "C05", "target": "C06", "dependency_strength": 0.85},
    {"source": "C06", "target": "C07", "dependency_strength": 0.95},
    {"source": "C07", "target": "C08", "dependency_strength": 0.90},
    {"source": "C07", "target": "C09", "dependency_strength": 0.95},
    {"source": "C09", "target": "C10", "dependency_strength": 0.85},
    {"source": "C04", "target": "C11", "dependency_strength": 0.75},
    {"source": "C05", "target": "C12", "dependency_strength": 0.70},
    {"source": "C03", "target": "C07", "dependency_strength": 0.60}
])

prerequisites.to_csv("data/prerequisites.csv", index=False)

# ---------------------------------------------------------
# 3. Student profile dataset
# ---------------------------------------------------------

students = pd.DataFrame([
    {
        "student_id": "S001",
        "student_name": "Aarav",
        "prior_level": "Beginner",
        "preferred_difficulty": 2,
        "learning_style": "Video",
        "daily_time_budget_min": 90,
        "target_mastery": 0.80,
        "motivation_level": 0.78
    },
    {
        "student_id": "S002",
        "student_name": "Maya",
        "prior_level": "Intermediate",
        "preferred_difficulty": 3,
        "learning_style": "Practice",
        "daily_time_budget_min": 120,
        "target_mastery": 0.85,
        "motivation_level": 0.84
    },
    {
        "student_id": "S003",
        "student_name": "Omar",
        "prior_level": "Beginner",
        "preferred_difficulty": 2,
        "learning_style": "Interactive",
        "daily_time_budget_min": 75,
        "target_mastery": 0.75,
        "motivation_level": 0.70
    },
    {
        "student_id": "S004",
        "student_name": "Sara",
        "prior_level": "Advanced",
        "preferred_difficulty": 4,
        "learning_style": "Coding",
        "daily_time_budget_min": 150,
        "target_mastery": 0.90,
        "motivation_level": 0.88
    },
    {
        "student_id": "S005",
        "student_name": "Rohan",
        "prior_level": "Intermediate",
        "preferred_difficulty": 3,
        "learning_style": "Reading",
        "daily_time_budget_min": 100,
        "target_mastery": 0.82,
        "motivation_level": 0.76
    }
])

students.to_csv("data/students.csv", index=False)

# ---------------------------------------------------------
# 4. Initial mastery scores
# One row per student-concept pair
# ---------------------------------------------------------

level_ranges = {
    "Beginner": (0.05, 0.35),
    "Intermediate": (0.25, 0.60),
    "Advanced": (0.45, 0.80)
}

mastery_rows = []

for _, student in students.iterrows():
    low, high = level_ranges[student["prior_level"]]

    for _, concept in concepts.iterrows():
        base = np.random.uniform(low, high)

        # Easier concepts generally have slightly higher prior mastery.
        difficulty_adjustment = (5 - concept["difficulty"]) * 0.03

        # Advanced RL topics should be lower for most students.
        if concept["concept_id"] in ["C07", "C08", "C09", "C10"]:
            base -= 0.08

        mastery = np.clip(base + difficulty_adjustment, 0.02, 0.95)

        mastery_rows.append({
            "student_id": student["student_id"],
            "concept_id": concept["concept_id"],
            "initial_mastery": round(float(mastery), 2)
        })

initial_mastery = pd.DataFrame(mastery_rows)
initial_mastery.to_csv("data/initial_mastery.csv", index=False)

# ---------------------------------------------------------
# 5. Historical quiz/performance data
# This simulates previous attempts and learning outcomes.
# ---------------------------------------------------------

quiz_rows = []

for _, student in students.iterrows():
    attempted_concepts = np.random.choice(concepts["concept_id"], size=6, replace=False)

    for attempt_no, concept_id in enumerate(attempted_concepts, start=1):
        concept = concepts[concepts["concept_id"] == concept_id].iloc[0]
        current_mastery = initial_mastery[
            (initial_mastery["student_id"] == student["student_id"]) &
            (initial_mastery["concept_id"] == concept_id)
        ]["initial_mastery"].values[0]

        score = np.clip(
            45 + current_mastery * 45 - concept["difficulty"] * 3 + np.random.normal(0, 8),
            20,
            98
        )

        time_spent = np.clip(
            concept["estimated_time_min"] + np.random.normal(0, 8),
            10,
            70
        )

        satisfaction = np.clip(
            0.55
            + 0.25 * student["motivation_level"]
            + 0.10 * concept["engagement_score"]
            - 0.04 * abs(student["preferred_difficulty"] - concept["difficulty"])
            + np.random.normal(0, 0.05),
            0.20,
            1.00
        )

        quiz_rows.append({
            "student_id": student["student_id"],
            "concept_id": concept_id,
            "attempt_no": attempt_no,
            "quiz_score": round(float(score), 1),
            "time_spent_min": round(float(time_spent), 1),
            "completed": 1 if score >= 50 else 0,
            "satisfaction_score": round(float(satisfaction), 2)
        })

quiz_history = pd.DataFrame(quiz_rows)
quiz_history.to_csv("data/quiz_history.csv", index=False)

# ---------------------------------------------------------
# 6. Learning objectives
# These are selectable goals in the app.
# ---------------------------------------------------------

learning_objectives = pd.DataFrame([
    {
        "objective_id": "O01",
        "objective_name": "Bayesian Reasoning Foundation",
        "target_skill_area": "Bayesian Reasoning",
        "required_concepts": "C01,C02,C03",
        "recommended_time_min": 90,
        "default_target_mastery": 0.80
    },
    {
        "objective_id": "O02",
        "objective_name": "Decision Theory Foundation",
        "target_skill_area": "Decision Theory",
        "required_concepts": "C04,C05",
        "recommended_time_min": 70,
        "default_target_mastery": 0.80
    },
    {
        "objective_id": "O03",
        "objective_name": "Reinforcement Learning Path",
        "target_skill_area": "Reinforcement Learning",
        "required_concepts": "C06,C07,C08,C09,C10",
        "recommended_time_min": 215,
        "default_target_mastery": 0.85
    },
    {
        "objective_id": "O04",
        "objective_name": "Exam MVP Preparation",
        "target_skill_area": "Mixed",
        "required_concepts": "C01,C04,C06,C07,C09,C10,C11,C12",
        "recommended_time_min": 300,
        "default_target_mastery": 0.82
    },
    {
        "objective_id": "O05",
        "objective_name": "Decision-Making Under Uncertainty Overview",
        "target_skill_area": "Mixed",
        "required_concepts": "C01,C02,C03,C04,C05,C06,C07,C11,C12",
        "recommended_time_min": 310,
        "default_target_mastery": 0.80
    }
])

learning_objectives.to_csv("data/learning_objectives.csv", index=False)

print("Synthetic data generated successfully in the data/ folder.")
print("Files created:")
print("- data/concepts.csv")
print("- data/prerequisites.csv")
print("- data/students.csv")
print("- data/initial_mastery.csv")
print("- data/quiz_history.csv")
print("- data/learning_objectives.csv")