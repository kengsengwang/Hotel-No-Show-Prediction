# 🏨 Hotel No-Show Prediction

This project aims to predict whether a hotel customer will show up or not, using various booking and demographic features. The model helps hotel management reduce costs due to no-shows by enabling preemptive measures such as overbooking or targeted confirmations.

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA process included:

### 🧹 Data Cleaning
- Removed duplicates.
- Handled missing values (especially in `children`, `agent`, and `country`).
- Dropped irrelevant or leakage columns like `reservation_status_date`.

### 📊 Key Observations
- **No-show rate**: ~37% of bookings resulted in no-shows.
- **Important features**:
  - Longer `lead_time` correlates with higher no-show probability.
  - Bookings made through certain channels (e.g., `agent`) have different no-show patterns.
  - Repeated guests and those with previous cancellations tend to no-show more.
- **Imbalanced target**: The `is_canceled` column is skewed, requiring balancing techniques.

### 📈 Visualizations
- Correlation heatmap.
- Count plots for categorical variables (`deposit_type`, `distribution_channel`).
- Boxplots showing how `lead_time`, `adr`, and `booking_changes` differ between show and no-show classes.

---

## 🧠 Machine Learning Pipeline

Implemented models:
- Logistic Regression
- Random Forest
- Deep Neural Network (Keras)

Pipeline steps:
1. Data loading from `data/hotel_no_show_cleaned.csv`
2. Preprocessing:
   - Label encoding for binary features
   - One-hot encoding for multi-class categorical variables
   - Scaling numeric features
3. Model training
4. Evaluation with accuracy, recall, precision, ROC-AUC

---

## 📁 Folder Structure

