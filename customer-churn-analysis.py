from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges").cast("double")))

    categorical_cols = ["gender", "PhoneService", "InternetService"]
    indexers = [StringIndexer(inputCol=col, outputCol=col + "Index") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "Index", outputCol=col + "Vec") for col in categorical_cols]

    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
    
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    encoded_features = [col + "Vec" for col in categorical_cols]
    feature_cols = numeric_cols + encoded_features

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    from pyspark.ml import Pipeline
    stages = indexers + encoders + [label_indexer, assembler]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    final_df = model.transform(df)
    return final_df.select("features", "label")

# Task 2: Train and Evaluate Logistic Regression Model
def train_logistic_regression_model(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression()
    model = lr.fit(train_df)
    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"Logistic Regression AUC: {auc:.4f}")

# Task 3: Feature Selection using Chi-Square Test
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    selected_df = selector.fit(df).transform(df)
    print("Top 5 selected features:")
    selected_df.select("selectedFeatures", "label").show(5, truncate=False)

# Task 4: Hyperparameter Tuning and Model Comparison
def tune_and_compare_models(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GBTClassifier": GBTClassifier()
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [3, 5]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].numTrees, [10, 20]).build(),
        "GBTClassifier": ParamGridBuilder().addGrid(models["GBTClassifier"].maxIter, [10, 20]).build(),
    }

    best_auc = 0.0
    best_model_name = ""
    best_model = None

    for name, model in models.items():
        print(f"Tuning {name}...")
        cv = CrossValidator(estimator=model,
                            estimatorParamMaps=param_grids[name],
                            evaluator=evaluator,
                            numFolds=5)
        cv_model = cv.fit(train_df)
        auc = evaluator.evaluate(cv_model.transform(test_df))
        print(f"{name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = cv_model.bestModel

    print(f"Best model: {best_model_name} with AUC = {best_auc:.4f}")

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)