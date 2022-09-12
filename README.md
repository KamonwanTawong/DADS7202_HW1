# DADS7202

## INTRODUCTION

## DATA
https://www.kaggle.com/datasets/adityakadiwal/water-potability

```
df = pd.read_csv('water_potability.csv')
df.head()
```

<img width="843" alt="ภาพถ่ายหน้าจอ 2565-09-12 เวลา 19 37 41" src="https://user-images.githubusercontent.com/107698198/189655445-3c1278fc-7661-4e34-8e30-d8c96c76b8e1.png">


The dataset contains 3276 rows and 10 columns

> 1. pH value: PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.

> 2. Hardness: Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.

> 3. Solids (Total dissolved solids - TDS): Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.

> 4. Chloramines: Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.

> 5. Sulfate: Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.

> 6. Conductivity: Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.

> 7. Organic_carbon: Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.

> 8. Trihalomethanes: THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.

> 9. Turbidity: The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.

> 10. Potability: Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.

```
df.info()
```
<img width="418" alt="ภาพถ่ายหน้าจอ 2565-09-12 เวลา 19 40 56" src="https://user-images.githubusercontent.com/107698198/189656535-c4e399f7-d871-4335-9293-c165d460bbee.png">


EXPLORATORY DATA ANALYSIS

```
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
df_clean = imp_mean.fit_transform(df)
```

```
df_clean = pd.DataFrame(df_clean,columns = df.columns)
```

```
df_clean.describe().style.background_gradient(cmap = "Reds")
```
<img width="935" alt="ภาพถ่ายหน้าจอ 2565-09-12 เวลา 19 48 02" src="https://user-images.githubusercontent.com/107698198/189657964-97dd91f7-382c-467a-bdaf-269c39836db7.png">


```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)
```

```
X = df_clean.drop('Potability',axis = 1)
y = df_clean.Potability
```

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## MACHINE LEARNING

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
```

```
models = [RandomForestClassifier(), KNeighborsClassifier(), SVC(), LogisticRegression() , MLPClassifier() , DecisionTreeClassifier()]
scores = dict()

for m in models:
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    print(f'model: {str(m)}')
    print(classification_report(y_test,y_pred, zero_division=1))
    print('-'*30, '\n')
```
<img width="506" alt="ภาพถ่ายหน้าจอ 2565-09-12 เวลา 21 46 02" src="https://user-images.githubusercontent.com/107698198/189684562-27c249fe-302e-4902-876e-01096f89956c.png">
<img width="533" alt="ภาพถ่ายหน้าจอ 2565-09-12 เวลา 21 50 02" src="https://user-images.githubusercontent.com/107698198/189685512-8a9d51f8-7b59-462b-856a-004300ddd99b.png">
<img width="496" alt="ภาพถ่ายหน้าจอ 2565-09-12 เวลา 20 13 12" src="https://user-images.githubusercontent.com/107698198/189663573-d345f3a5-5907-4fe7-a851-fde06bcd9d37.png">

## NETWORK ARCHITECTURE 

```
input_dim = 9     # the number of features per one input
output_dim = 2     # the number of output classes

model = tf.keras.models.Sequential()

# Input layer
model.add( tf.keras.Input(shape=(input_dim,)) )

# Hidden layer
model.add( tf.keras.layers.Dense(32, activation='relu', name='hidden1') )   # use default weight initialization, don't use any regularization
model.add( tf.keras.layers.BatchNormalization(axis=-1, name='bn1') )  
model.add( tf.keras.layers.Dense(64, activation='relu', name='hidden2') )   # use default weight initialization, don't use any regularization
model.add( tf.keras.layers.BatchNormalization(axis=-1, name='bn2') )
model.add( tf.keras.layers.Dense(32, activation='relu', name='hidden3') )   # use default weight initialization, don't use any regularization
model.add( tf.keras.layers.Dropout(0.3) )                        # drop rate = 30%

# Output layer
model.add( tf.keras.layers.Dense(output_dim, activation='softmax', name='output') )

model.summary()
```

<img width="581" alt="ภาพถ่ายหน้าจอ 2565-09-12 เวลา 20 25 53" src="https://user-images.githubusercontent.com/107698198/189666211-625476db-2680-44ca-a62f-b13a5c4bb424.png">


## COMPILE THE MODEL 

```
# Set fixed seeding values for reproducability during experiments
# Skip this cell if random initialization (with varied results) is needed
np.random.seed(1234)
tf.random.set_seed(5678)
```

```
# Compile with default values for both optimizer and loss
model.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'] )
```

```
# Compile + hyperparameter tuning
model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.001) , 
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) ,
                       metrics=['acc'] )
 ```
 
 ```
 checkpoint_filepath = "bestmodel_epoch{epoch:02d}_valloss{val_loss:.2f}.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath,
                                                                                              save_weights_only=True,
                                                                                              monitor='val_acc',
                                                                                              mode='max',
                                                                                              save_best_only=True)
 ```
 
 ```
 history = model.fit ( x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2, callbacks=[model_checkpoint_callback] )
 ```
 Iteration
 > + **Iteration 1** <br />
 np.random.seed(1234) <br />
 tf.random.set_seed(5678) <br />
 batch_size = 128 <br />
 epochs = 20 <br />
 training time per epoch = 0.05s <br />
 accuracy = 0.7299 <br />
 loss =  0.5378 <br />
 <img width="1009" alt="ภาพถ่ายหน้าจอ 2565-09-12 เวลา 20 32 19" src="https://user-images.githubusercontent.com/107698198/189667635-dda34815-f5ec-4ad1-b110-0c6a0c92afc5.png">
 
 
 
  > + **Iteration 2** <br />
 np.random.seed(0) <br />
 tf.random.set_seed(0) <br />
 batch_size = 128 <br />
 epochs = 20 <br />
 training time per epoch = 0.1s <br />
 accuracy = 0.7157 <br />
 loss = 0.5539 <br />
 ![messageImage_1662998773588](https://user-images.githubusercontent.com/107698198/189702724-2f3b309b-818b-419b-93c9-97bf7c71b74b.jpg)

 
  > + **Iteration 3** <br />
 np.random.seed(5678) <br />
 tf.random.set_seed(1234) <br />
 batch_size = 128 <br />
 epochs = 20 <br />
 training time per epoch = 0.05s <br />
 accuracy = <br />
 loss = <br />
 
  > + **Iteration 4** <br />
 np.random.seed(1234) <br />
 tf.random.set_seed(5678) <br />
 batch_size = 128 <br />
 epochs = 20 <br />
 training time per epoch = 0.05s <br />
 accuracy = <br />
 loss = <br />
 
  > + **Iteration 5** <br />
 np.random.seed(1234) <br />
 tf.random.set_seed(5678) <br />
 batch_size = 128 <br />
 epochs = 20 <br />
 training time per epoch = 0.05s <br />
 accuracy = <br />
 loss = <br />
 








