import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import numpy as np
import zipfile
import os
from PIL import Image
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle 

# Anna Schäfer 
# Matriculation number: 6755175 

st.title("Application for ML Tasks")
st.caption("This application allows you to upload a dataset (of a specific resolution: :blue[csv, tsv, xlsx, xml, json, npz, zip]), perform a small set of preprocessing procedures, and train Machine Learning Models depending on the type of data being uploaded: :blue[Tabular Data, Images, Texts].")
st.caption("This App is created by Anna Schäfer.")
st.caption("Images for design are created by ChatGPT.")

image_path = "ssss.jpeg"
image = Image.open(image_path)
st.image(image)

# FUNCTIONS

# DF
# this function takes the file and return a data frame 
# + possible contained images/texts and possible data type depending on data resolution
def return_df(file):
    name = file.name
    extension = name.split(".")[-1]

    if extension == "csv":
        df = pd.read_csv(file)
        data_type = "Tabular Data"
    elif extension == "tsv":
        df = pd.read_csv(file, sep="\t")
        data_type = "Tabular Data"
    elif extension == "xlsx":
        df = pd.read_excel(file)
        data_type = "Tabular Data"
    elif extension == "xml":
        df = pd.read_xml(file)
        data_type = "Tabular Data"
    elif extension == "json":
        df = pd.read_json(file)
        data_type = "Tabular Data"
    elif extension == "npz":
        images, data_type, structure = npz_check(file)
        if data_type == "Image Data":
            return images,None, None, data_type, structure
    elif extension == "zip":
        df, img_data, txt_data, data_type, structure = zip_check(file)
        return df, img_data, txt_data, data_type, structure
    else:
        st.error("Unsupported File Type")
        df = None
        data_type = None
    return df, None, None, data_type, None

# NPZ
# this function process the NPZ files, considering only images containing
def npz_check(file):
    npz_file = np.load(file)
    img_arr = [npz_file[key] for key in npz_file.files]

    if all(isinstance(arr, np.ndarray) for arr in img_arr):
        return img_arr, "Image Data", npz_file
    else:
        return None, "Unknown Data Type", None

# ZIP
# this function process the ZIP files, considering tables/images/texts containing
def zip_check(file):
    with zipfile.ZipFile(file, "r") as z:
        z.extractall("temporary_dir")
        extr_files = os.listdir("temporary_dir")
        tab_data = False
        img_data = False
        txt_data = False
        structure = extr_files

        df = None
        images = []
        texts = []

        for extr_file in extr_files:
            extension = extr_file.split(".")[-1]
            path = os.path.join("temporary_dir", extr_file)

            if extension in ["csv", "tsv", "xlsx", "xml", "json"]:
                if extension == "csv":
                    df = pd.read_csv(path)
                elif extension == "tsv":
                    df = pd.read_csv(path, sep="\t")
                elif extension == "xlsx":
                    df = pd.read_excel(path)
                elif extension == "xml":
                    df = pd.read_xml(path)
                elif extension == "json":
                    df = pd.read_json(path)
                tab_data = True

            elif extension in ["png", "jpg", "jpeg", "tiff"]:
                images.append(path)
                img_data = True

            elif extension == "txt":
                with open(path, "r") as file:
                    texts.append(file.read())
                txt_data = True

        if tab_data:
            data_type = "Tabular Data"
        elif img_data:
            data_type = "Image Data"
        elif txt_data:
            data_type = "Text Data"
        else:
            data_type = "Unknown Data Type"
        return df, images, texts, data_type, structure
    
# SAVE NEW DF
# this function save a new transformed Data Frame into .csv
def save_new_df(df, file_name):
    df.to_csv(file_name, index=False)
    st.success(f"Dataset has been saved as {file_name}")

# CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
        
st.info("Please upload Your file here to start:")
f = st.file_uploader("Please Upload the dataset", type=["csv", "tsv", "xlsx", "xml", "json", "npz", "zip"])

if f:
    df, img_data, txt_data, data_type, structure = return_df(f)
    if "df" not in st.session_state:
        st.session_state.df = df
    df = st.session_state.df

# APPLICATION DETECT THE DATA TYPE (HOPEFULLY CORRECT FOR THE LARGE PART OF DATA) 
# IN THE FRAMEWORK OF OUR COURSE AND THE PORPOSED DATA EXAMPLES

# TABULAR
    if data_type == "Tabular Data" and df is not None:
        st.success("File is Uploaded! Probable Data Type: Tabular Data")
        st.subheader("Data Exploring")
        st.info("Here You can see Your Data Frame and explore the structure of the file:")
        if st.button("Click here to see the Data Frame"):
            st.dataframe(df)
        st.session_state.df = df
        with st.container(height=700):
            tab1, tab2 = st.tabs(["EDA", "3D Visualization"])
            with tab1:
                st.write("Exploratory Data Analysis(EDA)")
                with st.form("Show Data Profiling"):
                   if st.form_submit_button("Show the Data Profiling"):
                       pr = ProfileReport(df)
                       st_profile_report(pr)
            with tab2:
                st.write("Select 3 Features and Your Target Variable")
                with st.form("Select 3 Features and Your Target"):
                    col_fea1, col_fea2, col_fea3, col_target = st.columns(4)
                    with col_fea1:
                       fea_1 = st.selectbox("Please select the 1st feature", df.columns)
                    with col_fea2:
                        fea_2 = st.selectbox("Please select the 2nd feature", df.columns)
                    with col_fea3:
                        fea_3 = st.selectbox("Please select the 3rd feature", df.columns)
                    with col_target:
                        target = st.selectbox("Please select the target label", df.columns)
                    if st.form_submit_button("Show the Visualization"): 
                       fig_3d = px.scatter_3d(df, x=fea_1, y=fea_2, z=fea_3, color=target)
                       st.plotly_chart(fig_3d)
        

        st.header("Put the Data to the Proper Form")
        st.info("Now You need to preprocess Your Data set before starting with model training.")
        st.subheader("Checking of Missing and Zero Values")
        with st.form("Checking of Missing and Zero Values"):
            if st.form_submit_button("Check the Data"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info("Hopfully, Your Data Set is full :)")
                    missing = df.isnull().sum()
                    st.write("Missing Values in each Column")
                    st.write(missing)


                with col2:
                    st.warning("Be careful! Usually Zero values are normal")
                    zero = (df == 0).sum()
                    st.write("Zero Values in each Column")
                    st.write(zero)

                with col3:
                    st.warning("Check unique values to be sure")
                    uniq = df.nunique()
                    st.write("Count of Unique Values")
                    st.write(uniq)

        missing = df.isnull().sum()
        zero = (df == 0).sum()
        col_miss = missing[missing > 0].index.tolist()
        col_zero = zero[zero > 0].index.tolist()

        st.subheader("Handle Missing and Zero Values")
                
        col_1, col_2 = st.columns(2)
        with col_1:
            with st.container(height=500, border=True):
                st.write("Decide what to do with Missing Values:")
            
                for col in col_miss:
                    with st.form(f"Handle Missing Values in: {col}"):
                        st.write(f"Column: {col}")
                        handle_m = st.selectbox(f"Select action for Missing Values in {col}:",
                                                    ["Do nothing", "Drop rows", "Replace with mean", "Replace with median"],
                                                    key=f"missing_{col}")
                
                        if st.form_submit_button(f"Handle Missing Values in: {col}"):
                            if handle_m == "Drop rows":
                                st.session_state.df = st.session_state.df.dropna(subset=[col])
                                st.success("Rows with Missing values dropped")
                            elif handle_m == "Replace with mean":
                                st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mean())
                                st.success("Missing Valuas are replaced with mean value")
                            elif handle_m == "Replace with median":
                                st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mean())
                                st.success("Missing Valuas are replaced with median value")
                            else:
                                st.write("No Missing Values found") 
                             

        with col_2:
            with st.container(height=500, border=True):
                st.write("Decide what to do with Zero Values:")
                for col in col_zero:
                    with st.form(f"Handle Zero Values in: {col}"):
                        st.write(f"Column: {col}")
                        handle_z = st.selectbox(f"Select action for Zero Values in: {col}:",
                                                    ["Do nothing", "Drop rows", "Replace with mean", "Replace with median"],
                                                    key=f"zero_{col}")
            
                        if st.form_submit_button(f"Handle Zero Values in: {col}"):
                            if handle_z == "Drop rows":
                                st.session_state.df = st.session_state.df[st.session_state.df[col] != 0]
                                st.success("Rows with Zero values dropped")
                            elif handle_z == "Replace with mean":
                                st.session_state.df[col] = st.session_state.df[col].replace(0, st.session_state.df[col].mean())
                                st.success("Zero Values are replaced with mean value")
                            elif handle_z == "Replace with median":
                                st.session_state.df[col] = st.session_state.df[col].replace(0, st.session_state.df[col].mean())
                                st.success("Zero Values are replaced with median value")
                            else:
                                st.write("No Zero Values found") 

        st.subheader("Convert Categorical Features to Numeric") 
        with st.form("Convert Categorical Features to Numeric"): 
            cat_columns = df.select_dtypes(include=['object']).columns.tolist()
            if cat_columns: 
                col_select = st.multiselect("Select Categorical Columns to Convert", cat_columns, placeholder="Choose some/all options",)
                if st.form_submit_button("Apply Label Encoding"):
                    for col in col_select:
                        le = LabelEncoder()
                        st.session_state.df[col] = le.fit_transform(st.session_state.df[col]) 
                    else:
                        st.write("No categorical columns found anymore")
                    st.success(f"Columns have been converted using Label Encoding.") 

        st.info("Let's have a look at our preprocessed Data:")
        if st.button("Show Preprocessed Data Frame"):
            st.dataframe(st.session_state.df)

        st.subheader ("Download Preprocessed Data")
        @st.cache_data
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")

        csv = convert_df(st.session_state.df)

        st.download_button(label="Download data as CSV",data=csv,file_name="preprocessed_dataset.csv",mime="text/csv",)

        st.header("Machine Learning Step")
        st.subheader("Parameters")
        with st.form("Please select Your target variable"):
            target = st.selectbox("Please select Your target variable", st.session_state.df.columns)
            if st.form_submit_button("Select"):
                y=st.session_state.df[target]
                st.success("Done!")

        #st.info("Here is the pairplot that can help to choose the features")        
        #if st.button("Show the PairPlots") :       
                #plot = sns.pairplot(st.session_state.df, hue=target, diag_kind="hist")
                #st.pyplot(plot)

        with st.form("Please select features and other parameters"):
            st.info(" Now choose features that make sence and don't choose the target variable:")
            variables = st.multiselect("Please select features", st.session_state.df.columns.tolist(),placeholder="Choose some options")
            task_type = st.selectbox("Please select the type of task:", ["Classification", "Regression"])
            st.warning("Be careful! A large test set can be pointless")
            test_s = st.slider("Select a size of test set", 0.0, 0.5, (0.25))

            if st.form_submit_button("Select"):
                y=st.session_state.df[target]
                X = st.session_state.df[variables]
                task = task_type
                test_s = test_s
                st.success(f"Done!") 
        

        st.subheader("Select the Models")
        st.info("Here you can choose models, which You want to train ")
        with st.form("Models"):
            y=st.session_state.df[target]
            X = st.session_state.df[variables]
            task = task_type
            test_s = test_s
            models_class = {"Random Forest Classifier":RandomForestClassifier(), 
                              "Logistic Regression": LogisticRegression(max_iter=2000), 
                              "SVC": SVC()
                              }
            models_reg = {"Random Forest Regressor":RandomForestRegressor(), 
                              "Linear Regression": LinearRegression(), 
                              "SVR": SVR()
                              }
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s, random_state=42)

            if task == "Classification":
                selected_models = st.multiselect("Please select the Classification Models", list(models_class.keys()), placeholder="Choose some/all options")
                models = {name:models_class[name] for name in selected_models}
                metric = accuracy_score
            else:
                selected_models = st.multiselect("Please select the Regression Models", list(models_reg.keys()))
                models = {name:models_reg[name] for name in selected_models}
                metric = mean_squared_error

            params = {"Random Forest Classifier": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
                          "Logistic Regression": {'C': [0.1, 1, 10]},
                          "SVC":{'C': [0.1, 1, 10], 'kernel':['linear', 'rbf']}, 
                          "Random Forest Regressor": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
                          "Linear Regression": {},
                          "SVR":{'C': [0.1, 1, 10], 'kernel':['linear', 'rbf']}
                          }
            results = {}
            
            if st.form_submit_button("Train"):
                for model_name, model in models.items():
                    st.write(f"Training {model_name}...")
                    grid = GridSearchCV(model, param_grid=params[model_name], cv=5)
                    grid.fit(X_train, y_train)
                    y_pred = grid.predict(X_test)

                    if task_type == "Classification":
                        score = accuracy_score(y_test, y_pred)
                    else:
                        score = mean_squared_error(y_test, y_pred, squared=False) # RMSE
                    
                    results[model_name] = (grid.best_params_, score)

            
                st.subheader("Results")
                st.success("Here You can see Best Parameters and Best Scores for each selected model")
                st.write("Results of Grid Search:")
                for model_name, (best_params, score) in results.items():
                    st.write("----------------------------------")
                    st.write(f"Model: {model_name}")
                    st.write(f"Best Parameters: {best_params}")
                    st.write(f"Score: {score}")

                st.success("Here You can see results of Accuracy Score(Classification) or RMSE(Regression) of the Best Model")
                best_model_name = max(results, key=lambda x: results[x][1] if task_type == "Classification" else -results[x][1])
                st.write(f"The best model: {best_model_name}, with a score of {results[best_model_name][1]}")

# IMAGING
    elif data_type == "Image Data":
        data = structure
        st.success("File is Uploaded! Probable Data Type: Image Data")
        st.subheader("Data Exploring")
        st.info("Here You can see some exsamples of the data fron Yuor data set and explore the structure of the file")
        with st.container(height=450):
            if st.button("Click here to see the Data"):
                if isinstance(df, list):
                    for i, img_arr in enumerate(df[:1]):
                        if isinstance(img_arr, np.ndarray):
                            st.image(img_arr, use_column_width=False)
                        else:
                            st.error("Unsupported Image Format")
                        break
                else:
                    for i, img_path in enumerate(img_data[:1]):
                        img = Image.open(img_path)
                        st.image(img, caption=f"Example {i+1}: {os.path.basename(img_path)}", use_column_width=False)
        
        st.subheader("Training of Convolutional Neural Network(CNN)")
        st.info("Here You can set some parameters to start train CNN")
        with st.form("Set the Parameters"):
            col_1, col_2 = st.columns(2)
            with col_1:
                st.write("Structure of the .npz:")
                st.write(data.files)

                st.write("Augmentation Parameters:")
                rotation_range = st.slider("Rotation range", 0, 360, (20))
                width_shift_range = st.slider("Width shift range", -1.0, 1.0, (0.2))
                height_shift_range = st.slider("Hight shift range", -1.0, 1.0, (0.2))

            with col_2:
                st.write("Test-Train-Validation Separation:")
                training = st.text_input("Training set:", "train_images")
                training_l = st.text_input("Training Labels:", "train_labels")
                validation = st.text_input("Validation set:", "val_images")
                validation_l = st.text_input("Validation Labels:", "val_labels")
                testing = st.text_input("Testing set:", "test_images")
                testing_l = st.text_input("Testing Labels:", "test_labels")

                st.write("Set the number of Epochs:")
                epochs = st.slider("Epochs", 1, 300, (10))
                st.write("Set the value of Patience-Parameter:")
                st.warning("Be careful! Patience must be less then number of Epochs")
                patience = st.slider("Patience", 1, 10, (2))

            if st.form_submit_button("Set the Parameters"):
                train_images = np.array(data[f'{training}'])
                train_labels = np.array(data[f'{training_l}'])
                val_images = np.array(data[f'{validation}'])
                val_labels = np.array(data[f'{validation_l}'])
                test_images = np.array(data[f'{testing}'])
                test_labels = np.array(data[f'{testing_l}'])
                epochs = epochs
                patience = patience
                rotation_range = rotation_range
                width_shift_range = width_shift_range
                height_shift_range = height_shift_range
                st.success("Done!")


        st.subheader("Results")
        with st.form("Train CNN"):
            if st.form_submit_button("Train CNN"):
                col_1, col_2 = st.columns(2)
                with col_1:
                    train_images = np.array(data[f'{training}'])
                    train_labels = np.array(data[f'{training_l}'])
                    val_images = np.array(data[f'{validation}'])
                    val_labels = np.array(data[f'{validation_l}'])
                    test_images = np.array(data[f'{testing}'])
                    test_labels = np.array(data[f'{testing_l}'])
                    epochs = epochs

                    train_images = train_images / 255.0
                    val_images = val_images / 255.0
                    test_images = test_images / 255.0

                    num_classes = len(np.unique(train_labels))
                    train_labels = to_categorical(train_labels, num_classes)
                    val_labels = to_categorical(val_labels, num_classes)
                    test_labels = to_categorical(test_labels, num_classes)

                    train_images, train_labels = shuffle(train_images, train_labels)
                    val_images, val_labels = shuffle(val_images, val_labels)
                    test_images, test_labels = shuffle(test_images, test_labels)

                    input_shape = train_images.shape[1:]
                    model = create_cnn_model(input_shape, num_classes)
                    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))

                    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
                    st.write(f"Training without Callback:")
                    st.write(f"Test Accuracy: {test_accuracy}")

                    fig, ax = plt.subplots()
                    ax.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linestyle='-', marker='o')
                    ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='brown', linestyle='--', marker='x')
                    ax.set_title("Training and Validation Accuracy over Epochs")
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)


                    train_images = np.array(data[f'{training}'])
                    train_labels = np.array(data[f'{training_l}'])
                    val_images = np.array(data[f'{validation}'])
                    val_labels = np.array(data[f'{validation_l}'])
                    test_images = np.array(data[f'{testing}'])
                    test_labels = np.array(data[f'{testing_l}'])
                    epochs = epochs
                    rotation_range = rotation_range
                    width_shift_range = width_shift_range
                    height_shift_range = height_shift_range

            
                    datagen = ImageDataGenerator(
                        rotation_range=rotation_range,
                        width_shift_range=width_shift_range,
                        height_shift_range=height_shift_range,
                        horizontal_flip=True
                    )
            
                    train_aug = datagen.flow(train_images, train_labels, batch_size=len(train_images), shuffle=False)
                    aug_img, aug_labels = next(train_aug)
                    train_images = np.concatenate((train_images, aug_img), axis=0)
                    train_labels = np.concatenate((train_labels, aug_labels), axis=0)

                    train_images = train_images / 255.0
                    val_images = val_images / 255.0
                    test_images = test_images / 255.0

                    num_classes = len(np.unique(train_labels))
                    train_labels = to_categorical(train_labels, num_classes)
                    val_labels = to_categorical(val_labels, num_classes)
                    test_labels = to_categorical(test_labels, num_classes)

                    train_images, train_labels = shuffle(train_images, train_labels)
                    val_images, val_labels = shuffle(val_images, val_labels)
                    test_images, test_labels = shuffle(test_images, test_labels)

                    input_shape = train_images.shape[1:]
                    model = create_cnn_model(input_shape, num_classes)
                    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))

                    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
                    st.write(f"Training without Callback + Augmentation:")
                    st.write(f"Test Accuracy: {test_accuracy}")

                    fig, ax = plt.subplots()
                    ax.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linestyle='-', marker='o')
                    ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='brown', linestyle='--', marker='x')
                    ax.set_title("Training and Validation Accuracy over Epochs")
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)

                with col_2:
                    epochs = epochs
                    patience = patience
                    callback = keras.callbacks.EarlyStopping(monitor='loss',patience=patience, verbose=0)
                    train_images = np.array(data[f'{training}'])
                    train_labels = np.array(data[f'{training_l}'])
                    val_images = np.array(data[f'{validation}'])
                    val_labels = np.array(data[f'{validation_l}'])
                    test_images = np.array(data[f'{testing}'])
                    test_labels = np.array(data[f'{testing_l}'])
                    epochs = epochs
                    rotation_range = rotation_range
                    width_shift_range = width_shift_range
                    height_shift_range = height_shift_range

                    train_images = train_images / 255.0
                    val_images = val_images / 255.0
                    test_images = test_images / 255.0

                    num_classes = len(np.unique(train_labels))
                    train_labels = to_categorical(train_labels, num_classes)
                    val_labels = to_categorical(val_labels, num_classes)
                    test_labels = to_categorical(test_labels, num_classes)

                    train_images, train_labels = shuffle(train_images, train_labels)
                    val_images, val_labels = shuffle(val_images, val_labels)
                    test_images, test_labels = shuffle(test_images, test_labels)

                    input_shape = train_images.shape[1:]
                    model = create_cnn_model(input_shape, num_classes)
                    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels), callbacks=[callback])

                    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
                    st.write(f"Training with Callback:")
                    st.write(f"Test Accuracy: {test_accuracy}")

                    fig, ax = plt.subplots()
                    ax.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linestyle='-', marker='o')
                    ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='brown', linestyle='--', marker='x')
                    ax.set_title("Training and Validation Accuracy over Epochs")
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)

                    epochs = epochs
                    patience = patience
                    callback = keras.callbacks.EarlyStopping(monitor='loss',patience=patience, verbose=0)
                    train_images = np.array(data[f'{training}'])
                    train_labels = np.array(data[f'{training_l}'])
                    val_images = np.array(data[f'{validation}'])
                    val_labels = np.array(data[f'{validation_l}'])
                    test_images = np.array(data[f'{testing}'])
                    test_labels = np.array(data[f'{testing_l}'])
                    epochs = epochs
                    rotation_range = rotation_range
                    width_shift_range = width_shift_range
                    height_shift_range = height_shift_range

                    datagen = ImageDataGenerator(
                        rotation_range=rotation_range,
                        width_shift_range=width_shift_range,
                        height_shift_range=height_shift_range,
                        horizontal_flip=True
                    )
            
                    train_aug = datagen.flow(train_images, train_labels, batch_size=len(train_images), shuffle=False)
                    aug_img, aug_labels = next(train_aug)
                    train_images = np.concatenate((train_images, aug_img), axis=0)
                    train_labels = np.concatenate((train_labels, aug_labels), axis=0)

                    train_images = train_images / 255.0
                    val_images = val_images / 255.0
                    test_images = test_images / 255.0

                    num_classes = len(np.unique(train_labels))
                    train_labels = to_categorical(train_labels, num_classes)
                    val_labels = to_categorical(val_labels, num_classes)
                    test_labels = to_categorical(test_labels, num_classes)

                    train_images, train_labels = shuffle(train_images, train_labels)
                    val_images, val_labels = shuffle(val_images, val_labels)
                    test_images, test_labels = shuffle(test_images, test_labels)

                    input_shape = train_images.shape[1:]
                    model = create_cnn_model(input_shape, num_classes)
                    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels), callbacks=[callback])

                    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
                    st.write(f"Training with Callback + Augmentation:")
                    st.write(f"Test Accuracy: {test_accuracy}")

                    fig, ax = plt.subplots()
                    ax.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linestyle='-', marker='o')
                    ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='brown', linestyle='--', marker='x')
                    ax.set_title("Training and Validation Accuracy over Epochs")
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)

# TEXT
    elif data_type == "Text Data":
        st.success("File is Uploaded! Probable Data Type: Text Data")
        st.write("Data Demonstartion")
        with st.form("Text Demonstartion"):
            if st.form_submit_button("Show the Data"):
                for i, text in enumerate(txt_data[:2]):
                    st.text_area(f"Example {i+1}", text, height=300)

        st.header("Will be available in the next update")
        st.balloons()
        
    else:
        st.error("Unsupported or Unknown Data Type")
