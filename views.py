from django.shortcuts import render
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


def training(request):

    dataset_name = "EV_Battery_Lifetime_Dataset (1).csv"
    csv_path = os.path.join(settings.MEDIA_ROOT, dataset_name)

    if not os.path.exists(csv_path):
        return render(request, 'users/accuracy.html', {
            "results": "❌ Dataset Not Found in MEDIA Folder!"
        })

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    feature_cols = [
        "age_months",
        "odometer_km",
        "fast_charging_share",
        "avg_daily_km",
        "avg_temperature",
        "voltage_mean",
        "current_mean",
        "soh_percent"
    ]

    df["total_life_months"] = df["age_months"] + df["rul_months"]

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    SEQ_LEN = 1
    X, y = [], []

    df_sorted = df.sort_values(by=["vehicle_id", "age_months"])

    for vid, group in df_sorted.groupby("vehicle_id"):
        g = group.reset_index(drop=True)
        if len(g) >= SEQ_LEN:
            X.append(g.iloc[:SEQ_LEN][feature_cols].values)
            y.append(g["total_life_months"].max())

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        return render(request, 'users/accuracy.html', {
            "results": "❌ Not Enough Early Data!"
        })

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===== CNN + LSTM Hybrid Model =====
    inputs = layers.Input(shape=(SEQ_LEN, len(feature_cols)))
    x = layers.Conv1D(64, 1, activation="relu", padding="same")(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    algorithm_name = "CNN + LSTM Hybrid Deep Learning Model"

    es = EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    # Evaluation
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Save model
    model.save(os.path.join(settings.MEDIA_ROOT, "battery_cnn_lstm_model.h5"))

    # Loss Graph
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    loss_path = os.path.join(settings.MEDIA_ROOT, "loss_graph.png")
    plt.savefig(loss_path)
    plt.close()

    # Prediction Graph
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6, color='green')
    plt.xlabel("Actual Lifetime (months)")
    plt.ylabel("Predicted Lifetime (months)")
    plt.title("Actual vs Predicted Battery Lifetime")
    pred_path = os.path.join(settings.MEDIA_ROOT, "prediction_graph.png")
    plt.savefig(pred_path)
    plt.close()

    # Final Results Summary
    results_msg = f"""
        <b>Algorithm Used:</b><br>
        {algorithm_name}<br><br>
        <b>MAE:</b> {test_mae:.2f} months<br>
        <b>MSE:</b> {test_loss:.2f}<br>
        <b>R² Score Accuracy:</b> {r2:.2f} ✔<br>
        <b>Average Error:</b> {test_mae/12:.2f} years<br>
        <br>
        <strong>Training Status:</strong><br>
        🟢 Model Training Completed Successfully! 🚀
    """

    return render(request, 'users/accuracy.html', {
        "results": results_msg,
        "loss_graph": "/media/loss_graph.png",
        "prediction_graph": "/media/prediction_graph.png",
        "best_model": algorithm_name
    })


from django.shortcuts import render
import os
import numpy as np
import pandas as pd
import joblib
from django.conf import settings
from tensorflow.keras.models import load_model

# Feature list (must match training)
feature_cols = [
    "age_months",
    "odometer_km",
    "fast_charging_share",
    "avg_daily_km",
    "avg_temperature",
    "voltage_mean",
    "current_mean",
    "soh_percent"
]

def prediction(request):
    result = None

    if request.method == "POST":
        # Collect form data
        user_input = [
            float(request.POST.get(col)) for col in feature_cols
        ]

        # Load saved model & scaler
        model_path = os.path.join(settings.MEDIA_ROOT, "battery_cnn_lstm_model.h5")
        model = load_model(model_path)

        csv_path = os.path.join(settings.MEDIA_ROOT, "EV_Battery_Lifetime_Dataset (1).csv")
        df = pd.read_csv(csv_path)
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        # Preprocess input
        arr = np.array(user_input).reshape(1, 1, len(feature_cols))
        arr_scaled = scaler.transform(arr.reshape(1, len(feature_cols))).reshape(1, 1, len(feature_cols))

        # Prediction
        total_life_pred = model.predict(arr_scaled)[0][0]  # in months
        age_now = user_input[0]  # age_months entered by user
        remaining_months = total_life_pred - age_now

        # Years conversion
        remaining_years = remaining_months / 12

        # Battery SOH interpretation
        soh = user_input[7]
        if soh >= 0.85:
            health = "Excellent 🚗⚡"
        elif soh >= 0.70:
            health = "Good 🙂"
        else:
            health = "Poor ⚠️ Battery Needs Check"

        result = {
            "total_months": round(total_life_pred, 2),
            "remaining_months": round(remaining_months, 2),
            "remaining_years": round(remaining_years, 2),
            "health_status": health
        }

    return render(request, "users/prediction.html", {"result": result, "features": feature_cols})



# Create your views here.
import os
def ViewDataset(request):
    csv_path = os.path.join(settings.MEDIA_ROOT, 'EV_Battery_Lifetime_Dataset (1).csv')

    if not os.path.exists(csv_path):
        return render(request, 'users/viewData.html', {"data": "<p>Dataset not found!</p>"})

    df = pd.read_csv(csv_path)

    # Show up to 200 rows for good viewing performance
    df_html = df.head(200).to_html(
        index=False,
        classes="styled-table",
        justify="center",
        border=0
    )

    return render(request, 'users/viewData.html', {
        'data': df_html,
        'title': "EV Battery Dataset"
    })



from django.shortcuts import render, redirect
from .models import UserRegistrationModel
from django.contrib import messages

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request,"Registration successful!")
    return render(request, 'UserRegistrations.html') 


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def index(request):
    return render(request,"index.html")
