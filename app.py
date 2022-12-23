#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:





# In[ ]:


import streamlit as st 
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout='wide',initial_sidebar_state='expanded')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

header =st.container()
dataset = st.container()
features = st.container()
model_training=st.container()

with header:
    st.title('Streamlit GUI')

with dataset:
    uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"])

# Check if file was uploaded
if uploaded_file:
    # Check MIME type of the uploaded file
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Work with the dataframe
    

  
    
   # df = pd.read_csv('/Users/bunnysair/Desktop/archive 2/PJME_hourly.csv')
    

# if  dataset ="PJMW_MW"
   
# else  if dataset = "PJME_MW"
#     df = pd.read_csv('/Users/bunnysair/Desktop/archive 2/PJME_hourly.csv')

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components

    with features:
        import plotly.graph_objs as go
        from plotly.offline import iplot

        def plot_dataset(df, title):
            data = []
            
            value = go.Scatter(
                x=df.index,
                y=df.value,
                mode="lines",
                name="values",
                marker=dict(),
                text=df.index,
                line=dict(color="rgba(128,0,128, 0.3)"),
            )
            data.append(value)

            layout = dict(
                title=title,
                xaxis=dict(title="Date", ticklen=5, zeroline=False),
                yaxis=dict(title="Value", ticklen=5, zeroline=False),
            )

            fig = dict(data=data, layout=layout)
            st.plotly_chart(fig, use_container_width=True)
        
            
        df = df.set_index(['Datetime'])
        df = df.rename(columns={'PJME_MW': 'value'})

        df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic:
            df = df.sort_index()
            
        plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')
        #value=st.selectbox('Pick one',['NONE','R2','Mean Absolute Error','Root Mean Squared Error'])
    def generate_time_lags(df, n_lags):
        df_n = df.copy()
        for n in range(1, n_lags + 1):
            df_n[f"lag{n}"] = df_n["value"].shift(n)
        df_n = df_n.iloc[n_lags:]
        return df_n

    input_dim = 100

    df_timelags = generate_time_lags(df, input_dim)

    df_features = (
                    df
                    .assign(hour = df.index.hour)
                    .assign(day = df.index.day)
                    .assign(month = df.index.month)
                    .assign(day_of_week = df.index.dayofweek)
                    .assign(week_of_year = df.index.week)
                )

    def onehot_encode_pd(df, cols):
        for col in cols:
            dummies = pd.get_dummies(df[col], prefix=col)
        
        return pd.concat([df, dummies], axis=1).drop(columns=cols)

    df_features = onehot_encode_pd(df_features, ['month','day','day_of_week','week_of_year'])
    #df_features

    def generate_cyclical_features(df, col_name, period, start_num=0):
        kwargs = {
            f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
            f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
                }
        return df.assign(**kwargs).drop(columns=[col_name])

    df_features = generate_cyclical_features(df_features, 'hour', 24, 0)
    # df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
    # df_features = generate_cyclical_features(df_features, 'month', 12, 1)
    # df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

    #df_features         
    #value=st.selectbox('Pick one',['NONE','R2','Mean Absolute Error','Root Mean Squared Error'])     
    metrics={'models':['GRU','LSTM'],'R2':[0.6130744407493718,0.49906310883365035],'Mean Absolute Error':[3018.3943,3654.771006607063],'Root Mean Squared Error':[4036.2994190223303,4592.627334481799]}
    value=st.selectbox('Pick one',['NONE','R2','Mean Absolute Error','Root Mean Squared Error'])
    if(value=='Root Mean Squared Error'):
            fig=px.bar(metrics,x='models',y='Root Mean Squared Error')
            st.plotly_chart(fig,use_container_width=True)
    elif(value=='Mean Absolute Error'):
            fig=px.bar(metrics,x='models',y='Mean Absolute Error')
            st.plotly_chart(fig,use_container_width=True)
    elif(value=='R2'):
            fig=px.bar(metrics,x='models',y='R2')
            st.plotly_chart(fig,use_container_width=True)
    from sklearn.model_selection import train_test_split

    def feature_label_split(df, target_col):
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    def train_val_test_split(df, target_col, test_ratio):
        val_ratio = test_ratio / (1 - test_ratio)
        X, y = feature_label_split(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
        return X_train, X_val, X_test, y_train, y_val, y_test

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.2)

    from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

    def get_scaler(scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    scaler = get_scaler('minmax')
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    from torch.utils.data import TensorDataset, DataLoader

    batch_size = 64

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    class LSTMModel(nn.Module):
        """LSTMModel class extends nn.Module class and works as a constructor for LSTMs.

        LSTMModel class initiates a LSTM module based on PyTorch's nn.Module class.
        It has only two methods, namely init() and forward(). While the init()
        method initiates the model with the given input parameters, the forward()
        method defines how the forward propagation needs to be calculated.
        Since PyTorch automatically defines back propagation, there is no need
        to define back propagation method.

        Attributes:
            hidden_dim (int): The number of nodes in each layer
            layer_dim (str): The number of layers in the network
            lstm (nn.LSTM): The LSTM model constructed with the input parameters.
            fc (nn.Linear): The fully connected layer to convert the final state
                            of LSTMs to our desired output shape.

        """
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
            """The __init__ method that initiates a LSTM instance.

            Args:
                input_dim (int): The number of nodes in the input layer
                hidden_dim (int): The number of nodes in each layer
                layer_dim (int): The number of layers in the network
                output_dim (int): The number of nodes in the output layer
                dropout_prob (float): The probability of nodes being dropped out

            """
            super(LSTMModel, self).__init__()

            # Defining the number of layers and the nodes in each layer
            self.hidden_dim = hidden_dim
            self.layer_dim = layer_dim

            # LSTM layers
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
            )

            # Fully connected layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            """The forward method takes input tensor x and does forward propagation

            Args:
                x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

            Returns:
                torch.Tensor: The output tensor of the shape (batch size, output_dim)

            """
            # Initializing hidden state for first input with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            # Initializing cell state for first input with zeros
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            # Forward propagation by passing in the input, hidden state, and cell state into the model
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
            # so that it can fit into the fully connected layer
            out = out[:, -1, :]

            # Convert the final state to our desired output shape (batch_size, output_dim)
            out = self.fc(out)

            return out
    class GRUModel(nn.Module):
        """GRUModel class extends nn.Module class and works as a constructor for GRUs.

        GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
        It has only two methods, namely init() and forward(). While the init()
        method initiates the model with the given input parameters, the forward()
        method defines how the forward propagation needs to be calculated.
        Since PyTorch automatically defines back propagation, there is no need
        to define back propagation method.

        Attributes:
            hidden_dim (int): The number of nodes in each layer
            layer_dim (str): The number of layers in the network
            gru (nn.GRU): The GRU model constructed with the input parameters.
            fc (nn.Linear): The fully connected layer to convert the final state
                            of GRUs to our desired output shape.

        """
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
            """The __init__ method that initiates a GRU instance.

            Args:
                input_dim (int): The number of nodes in the input layer
                hidden_dim (int): The number of nodes in each layer
                layer_dim (int): The number of layers in the network
                output_dim (int): The number of nodes in the output layer
                dropout_prob (float): The probability of nodes being dropped out

            """
            super(GRUModel, self).__init__()

            # Defining the number of layers and the nodes in each layer
            self.layer_dim = layer_dim
            self.hidden_dim = hidden_dim

            # GRU layers
            self.gru = nn.GRU(
                input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
            )

            # Fully connected layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            """The forward method takes input tensor x and does forward propagation

            Args:
                x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

            Returns:
                torch.Tensor: The output tensor of the shape (batch size, output_dim)

            """
            # Initializing hidden state for first input with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            # Forward propagation by passing in the input and hidden state into the model
            out, _ = self.gru(x, h0.detach())

            # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
            # so that it can fit into the fully connected layer
            out = out[:, -1, :]

            # Convert the final state to our desired output shape (batch_size, output_dim)
            out = self.fc(out)

            return out

    def get_model(model, model_params):
        models = {
        "gru": GRUModel,
        "lstm": LSTMModel,
        }
        return models.get(model.lower())(**model_params)

    class Optimization:
        """Optimization is a helper class that allows training, validation, prediction.

        Optimization is a helper class that takes model, loss function, optimizer function
        learning scheduler (optional), early stopping (optional) as inputs. In return, it
        provides a framework to train and validate the models, and to predict future values
        based on the models.

        Attributes:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
            train_losses (list[float]): The loss values from the training
            val_losses (list[float]): The loss values from the validation
            last_epoch (int): The number of epochs that the models is trained
        """
        def __init__(self, model, loss_fn, optimizer):
            """
            Args:
                model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
                loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
                optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
            """
            self.model = model
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.train_losses = []
            self.val_losses = []
            
        def train_step(self, x, y):
            """The method train_step completes one step of training.

            Given the features (x) and the target values (y) tensors, the method completes
            one step of the training. First, it activates the train mode to enable back prop.
            After generating predicted values (yhat) by doing forward propagation, it calculates
            the losses by using the loss function. Then, it computes the gradients by doing
            back propagation and updates the weights by calling step() function.

            Args:
                x (torch.Tensor): Tensor for features to train one step
                y (torch.Tensor): Tensor for target values to calculate losses

            """
            # Sets model to train mode
            self.model.train()

            # Makes predictions
            yhat = self.model(x)

            # Computes loss
            loss = self.loss_fn(y, yhat)

            # Computes gradients
            loss.backward()

            # Updates parameters and zeroes gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
            """The method train performs the model training

            The method takes DataLoaders for training and validation datasets, batch size for
            mini-batch training, number of epochs to train, and number of features as inputs.
            Then, it carries out the training by iteratively calling the method train_step for
            n_epochs times. If early stopping is enabled, then it  checks the stopping condition
            to decide whether the training needs to halt before n_epochs steps. Finally, it saves
            the model in a designated file path.

            Args:
                train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
                val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
                batch_size (int): Batch size for mini-batch training
                n_epochs (int): Number of epochs, i.e., train steps, to train
                n_features (int): Number of feature columns

            """
            model_path = f'{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

            for epoch in range(1, n_epochs + 1):
                batch_losses = []
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                    y_batch = y_batch.to(device)
                    loss = self.train_step(x_batch, y_batch)
                    batch_losses.append(loss)
                training_loss = np.mean(batch_losses)
                self.train_losses.append(training_loss)

                with torch.no_grad():
                    batch_val_losses = []
                    for x_val, y_val in val_loader:
                        x_val = x_val.view([batch_size, -1, n_features]).to(device)
                        y_val = y_val.to(device)
                        self.model.eval()
                        yhat = self.model(x_val)
                        val_loss = self.loss_fn(y_val, yhat).item()
                        batch_val_losses.append(val_loss)
                    validation_loss = np.mean(batch_val_losses)
                    self.val_losses.append(validation_loss)

                if (epoch <= 10) | (epoch % 50 == 0):
                    print(
                        f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                    )

            torch.save(self.model.state_dict(), model_path)

        def evaluate(self, test_loader, batch_size=1, n_features=1):
            """The method evaluate performs the model evaluation

            The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
            and number of features as inputs. Similar to the model validation, it iteratively
            predicts the target values and calculates losses. Then, it returns two lists that
            hold the predictions and the actual values.

            Note:
                This method assumes that the prediction from the previous step is available at
                the time of the prediction, and only does one-step prediction into the future.

            Args:
                test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
                batch_size (int): Batch size for mini-batch training
                n_features (int): Number of feature columns

            Returns:
                list[float]: The values predicted by the model
                list[float]: The actual values in the test set.

            """
            with torch.no_grad():
                predictions = []
                values = []
                for x_test, y_test in test_loader:
                    x_test = x_test.view([batch_size, -1, n_features]).to(device)
                    y_test = y_test.to(device)
                    self.model.eval()
                    yhat = self.model(x_test)
                    predictions.append(yhat.to(device).detach().numpy())
                    values.append(y_test.to(device).detach().numpy())

            return predictions, values

        def plot_losses(self):
            """The method plots the calculated loss values for training and validation
            """
            plt.plot(self.train_losses, label="Training loss")
            plt.plot(self.val_losses, label="Validation loss")
            plt.legend()
            plt.title("Losses")
            st.pyplot()
            plt.close()





    with model_training:
        
        import torch.optim as optim

        input_dim = len(X_train.columns)
        output_dim = 1
        hidden_dim = 64
        layer_dim = 3
        batch_size = 64
        dropout = 0.2
        n_epochs = 20
        learning_rate = 1e-3
        weight_decay = 1e-6

        model_params = {'input_dim': input_dim,
                        'hidden_dim' : hidden_dim,
                        'layer_dim' : layer_dim,
                        'output_dim' : output_dim,
                        'dropout_prob' : dropout}
        
        st.markdown("Models")
        col1,col2 = st.columns(2)
        with col1:
            st.write('LSTM')
            model = get_model('lstm', model_params)
            loss_fn = nn.MSELoss(reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
            opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
            opt.plot_losses()
            predictions, values = opt.evaluate(
            test_loader_one,
            batch_size=1,
            n_features=input_dim
            )
            def inverse_transform(scaler, df, columns):
                for col in columns:
                    df[col] = scaler.inverse_transform(df[col])
                return df    
            def format_predictions(predictions, values, df_test, scaler):
                vals = np.concatenate(values, axis=0).ravel()
                preds = np.concatenate(predictions, axis=0).ravel()
                df_result1 = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
                df_result1 = df_result1.sort_index()
                df_result1 = inverse_transform(scaler, df_result1, [["value", "prediction"]])
                return df_result1


            df_result1 = format_predictions(predictions, values, X_test, scaler)
            # df_result1   

        with col2:  
            st.write('GRU')   
            model = get_model('gru', model_params)
            loss_fn = nn.MSELoss(reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
            opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
            opt.plot_losses()

            predictions, values = opt.evaluate(
            test_loader_one,
            batch_size=1,
            n_features=input_dim
            )

            def inverse_transform(scaler, df, columns):
                for col in columns:
                    df[col] = scaler.inverse_transform(df[col])
                return df


            def format_predictions(predictions, values, df_test, scaler):
                vals = np.concatenate(values, axis=0).ravel()
                preds = np.concatenate(predictions, axis=0).ravel()
                df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
                df_result = df_result.sort_index()
                df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
                return df_result


            df_result = format_predictions(predictions, values, X_test, scaler)
            # df_result
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        with col2:
    
            def calculate_metrics(df):
                result_metrics = {'mae' : mean_absolute_error(df.value, df.prediction),
                                'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
                                'r2' : r2_score(df.value, df.prediction)
                                }
                st.write("Mean Absolute Error:       ", result_metrics["mae"])
                st.write("Root Mean Squared Error:   ", result_metrics["rmse"])
                st.write("R^2 Score:                 ", result_metrics["r2"])
                return result_metrics
            st.write('Metrics(GRU)')
            result_metrics = calculate_metrics(df_result)
        with col1:

            def calculate_metrics1(df):
                result_metrics1 = {'mae' : mean_absolute_error(df.value, df.prediction),
                                'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
                                'r2' : r2_score(df.value, df.prediction)}
                st.write("Mean Absolute Error:       ", result_metrics1["mae"])
                st.write("Root Mean Squared Error:   ", result_metrics1["rmse"])
                st.write("R^2 Score:                 ", result_metrics1["r2"])
                return result_metrics1
            st.write('Metrics(LSTM)')
            result_metrics1 = calculate_metrics(df_result1)
        
        


        from sklearn.linear_model import LinearRegression

        def build_baseline_model(df, test_ratio, target_col):
            X, y = feature_label_split(df, target_col)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, shuffle=False
            )
            model = LinearRegression()
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)

            result = pd.DataFrame(y_test)
            result["prediction"] = prediction
            result = result.sort_index()

            return result

        df_baseline = build_baseline_model(df_features, 0.2, 'value')
        #st.write('Linear Regression')
        #baseline_metrics = calculate_metrics(df_baseline)



        import plotly.graph_objs as go
        from plotly.offline import iplot


        def plot_predictions(df_result, df_baseline):
            data = []
            
            value = go.Scatter(
                x=df_result.index,
                y=df_result.value,
                mode="lines",
                name="values",
                marker=dict(),
                text=df_result.index,
                line=dict(color="rgba(0,128,0, 0.3)"),
            )
            data.append(value)
            
            prediction = go.Scatter(
                x=df_result.index,
                y=df_result.prediction,
                mode="lines",
                line={"dash": "dot"},
                name='predictions',
                marker=dict(),
                text=df_result.index,
                opacity=0.8,
            )
            data.append(prediction)
            
            layout = dict(
                title="Predictions vs Actual Values for the dataset",
                xaxis=dict(title="Time", ticklen=5, zeroline=False),
                yaxis=dict(title="Value", ticklen=5, zeroline=False),
            )

            fig = dict(data=data, layout=layout)
            st.plotly_chart(fig, use_container_width=True)

            
            
        # Set notebook mode to work in offline
        st.markdown("COMPARING MODELS USING PREDICTION V/S VALUES")
        col1,col2 = st.columns(2)
        with col1:
            st.write('GRU')
            plot_predictions(df_result, df_baseline)
        with col2:
            st.write("LSTM")
            plot_predictions(df_result1,df_baseline)





