import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os # Added for path handling

def create_encoders_for_adult_dataset(csv_file_path):
    """
    Create encoders specifically for the Adult/Census Income dataset
    """
    
    # Define column names based on the Adult dataset documentation
    # This is CRUCIAL because adult.data does not have a header row
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'educational_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]

    # Load the training data, specifying no header and providing column names
    print("Loading training data...")
    df = pd.read_csv(csv_file_path, header=None, names=column_names) 
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Apply the same preprocessing as your training pipeline
    print("\nApplying preprocessing...")
    
    # Strip whitespace from all string columns immediately after loading
    # This is critical because some values in the adult.data have leading spaces (e.g., ' Private')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
        
    # Handle missing values represented as '?'
    df = df.replace('?', 'Others')
    print(f"After replacing '?' with 'Others': {df.shape}")
    
    # Apply workclass filtering
    if 'workclass' in df.columns:
        original_count = len(df)
        df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
        print(f"After workclass filtering: {len(df)} (removed {original_count - len(df)} rows)")
    
    # Apply age filtering
    if 'age' in df.columns:
        original_count = len(df)
        df = df[(df['age'] <= 75) & (df['age'] >= 17)]
        print(f"After age filtering (17-75): {len(df)} (removed {original_count - len(df)} rows)")
    
    # Apply educational-num filtering
    if 'educational_num' in df.columns:
        original_count = len(df)
        df = df[(df['educational_num'] <= 16) & (df['educational_num'] >= 5)]
        print(f"After educational_num filtering (5-16): {len(df)} (removed {original_count - len(df)} rows)")
    
    # Drop education column (as mentioned in your API code)
    if 'education' in df.columns:
        df = df.drop(columns=['education'])
        print("Dropped 'education' column")
    
    print(f"Final preprocessed data shape: {df.shape}")
    
    # Define categorical features to encode (these names will now correctly match the DataFrame)
    categorical_features = [
        'workclass', 'marital_status', 'occupation', 
        'relationship', 'race', 'gender', 'native_country'
    ]
    
    # Create and fit encoders
    encoders = {}
    
    print("\nCreating encoders...")
    for feature in categorical_features:
        if feature in df.columns:
            print(f"\nProcessing {feature}:")
            
            # Check unique values
            unique_values = df[feature].unique()
            print(f"   Unique values: {len(unique_values)}")
            print(f"   Sample values: {unique_values[:10]}")
            
            # Create and fit encoder
            encoder = LabelEncoder()
            encoder.fit(df[feature].astype(str))
            encoders[feature] = encoder
            
            print(f"   Encoded classes: {encoder.classes_}")
            print(f"   Number of classes: {len(encoder.classes_)}")
            
            # Show encoding mapping
            print("   Encoding mapping:")
            for i, class_name in enumerate(encoder.classes_):
                print(f"     '{class_name}' -> {i}")
        else:
            print(f"Warning: {feature} not found in data (This warning should now disappear for these features!)")
    
    # Save encoders to pickle file
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print(f"\n✅ Encoders saved successfully to 'encoders.pkl'!")
    print(f"Available encoders: {list(encoders.keys())}")
    
    return encoders

def verify_encoders():
    """Verify that the encoders were saved correctly"""
    print("\n" + "="*50)
    print("VERIFYING ENCODERS")
    print("="*50)
    
    try:
        with open('encoders.pkl', 'rb') as f:
            loaded_encoders = pickle.load(f)
        
        print("✅ Encoders loaded successfully!")
        
        for feature, encoder in loaded_encoders.items():
            print(f"\n{feature.upper()}:")
            print(f"   Number of classes: {len(encoder.classes_)}")
            print(f"   Classes: {list(encoder.classes_)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading encoders: {e}")
        return False

def test_encoding_examples():
    """Test encoding with some example values"""
    print("\n" + "="*50)
    print("TESTING ENCODING EXAMPLES")
    print("="*50)
    
    try:
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Test examples based on your sample data
        # Note: These values should match the *stripped* values from the dataset
        test_cases = {
            'workclass': ['State-gov', 'Private', 'Self-emp-not-inc', 'Others'], # Added 'Others'
            'marital_status': ['Never-married', 'Married-civ-spouse', 'Divorced'],
            'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners'],
            'relationship': ['Not-in-family', 'Husband', 'Wife'],
            'race': ['White', 'Black'],
            'gender': ['Male', 'Female'],
            'native_country': ['United-States', 'Cuba', 'Others'] # Added 'Others'
        }
        
        for feature, test_values in test_cases.items():
            if feature in encoders:
                print(f"\n{feature}:")
                for value in test_values:
                    # Check if value exists in the encoder's classes directly
                    if value in encoders[feature].classes_:
                        encoded = encoders[feature].transform([value])[0]
                        print(f"   '{value}' -> {encoded}")
                    else:
                        print(f"   '{value}' -> NOT FOUND in encoder classes (would use 'Others' or first class in production)")
            else:
                print(f"\nSkipping {feature} - encoder not found.")
        
    except Exception as e:
        print(f"Error testing encoders: {e}")

if __name__ == "__main__":
    # Update this path to your actual training data CSV file
    csv_file_path = "./adult.data" # Using relative path to the current directory
    
    # This block handles finding the CSV file, either directly or via command line argument
    if not os.path.exists(csv_file_path):
        print(f"❌ Training data file not found: {csv_file_path}")
        print("\nPlease update the 'csv_file_path' variable with the correct path to your training CSV file.")
        print("\nAlternatively, you can run this script with:")
        print("python encoders.py your_actual_file.csv")
        
        import sys
        if len(sys.argv) > 1:
            csv_file_path = sys.argv[1]
            if os.path.exists(csv_file_path):
                print(f"Using file from command line: {csv_file_path}")
            else:
                print(f"File not found: {csv_file_path}")
                exit(1)
        else:
            exit(1)
    
    # Create encoders
    print("Creating encoders from Adult dataset...")
    encoders = create_encoders_for_adult_dataset(csv_file_path)
    
    # Verify encoders were saved correctly
    verify_encoders()
    
    # Test with example values
    test_encoding_examples()
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    print("1. Upload 'encoders.pkl' to your Render deployment (via Git push)")
    print("2. Make sure 'best_model.pkl' is also uploaded (via Git push)")
    print("3. Deploy your FastAPI application (Railway will do this on push)")
    print("4. Test the API endpoints")