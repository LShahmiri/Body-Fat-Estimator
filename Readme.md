## Dataset

The dataset used in this project is the **Body Fat Prediction Dataset**, which is publicly available on Kaggle.

🔗 Dataset source:  
https://www.kaggle.com/fedesoriano/body-fat-prediction-dataset

The dataset contains anthropometric measurements collected from adult male subjects.  
It includes body density measurements obtained through underwater weighing along with
multiple body circumference measurements.

The target variable is **Body Fat Percentage**, which can be derived from body density using **Siri's equation (1956)**.

Body Fat % = 495 / Density − 450


---

## Dataset Attributes

The dataset contains the following features:

| Feature | Description |
|------|------|
| Density | Body density determined from underwater weighing |
| BodyFat | Percent body fat calculated using Siri's equation |
| Age | Age of the subject (years) |
| Weight | Body weight (lbs) |
| Height | Height (inches) |
| Neck | Neck circumference (cm) |
| Chest | Chest circumference (cm) |
| Abdomen | Abdomen circumference (cm) |
| Hip | Hip circumference (cm) |
| Thigh | Thigh circumference (cm) |
| Knee | Knee circumference (cm) |
| Ankle | Ankle circumference (cm) |
| Biceps | Extended biceps circumference (cm) |
| Forearm | Forearm circumference (cm) |
| Wrist | Wrist circumference (cm) |

---

## Project Goal

The objective of this project is to build a **machine learning regression model**
that predicts body fat percentage using body measurements that are easy to obtain.

The trained model is deployed using **Flask** to provide a simple web interface
for estimating body fat percentage interactively.

---

## Disclaimer

This project is intended for **educational and demonstration purposes only**.
The predicted values should not be considered a medical diagnosis.
Commit changes
