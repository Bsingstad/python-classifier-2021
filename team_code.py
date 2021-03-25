#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import neurokit2 as nk
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'


################################################################################
#
# Training function
#
################################################################################
def find_R_peaks(ecg_data,samplefreq):
    try:
        _, rpeaks = nk.ecg_peaks(ecg_data, sampling_rate=samplefreq)
        r_peaks=rpeaks['ECG_R_Peaks']
        r_peaks = np.delete(r_peaks,np.where(np.isnan(r_peaks))[0]).astype(int)
    
    except:
        print("cleaning data")
        cleaned_ecg = nk.ecg_clean(ecg_data, sampling_rate=samplefreq, method="neurokit")
        try:
            _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=samplefreq)
            r_peaks=rpeaks['ECG_R_Peaks']
            r_peaks = np.delete(r_peaks,np.where(np.isnan(r_peaks))[0]).astype(int)
        except:
            print("could not analyse cleaned ECG")
            #Midlertidig løsning:
            r_peaks = np.array([0,1,2,3])
    return r_peaks

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')
    all_labels = []
    #classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        #classes |= set(get_labels(header))
        all_labels.append(get_labels(header))

    df_labels = pd.DataFrame(all_labels)

    SNOMED_scored=pd.read_csv("./dx_mapping_scored.csv", sep=",")
    SNOMED_unscored=pd.read_csv("./dx_mapping_unscored.csv", sep=",")
    for i in range(len(SNOMED_unscored.iloc[0:,1])):
        df_labels.replace(to_replace=str(SNOMED_unscored.iloc[i,1]), inplace=True ,value="undefined class", regex=True)

    one_hot = MultiLabelBinarizer()
    y_temp = one_hot.fit_transform(df_labels[0].str.split(pat=','))
    y_temp= np.delete(y_temp, -1, axis=1)
    classes = one_hot.classes_[0:-1]


    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    print("classes:",num_classes)


    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    data = np.zeros((num_recordings, 14), dtype=np.float32) # 6 features: 4 feature for based on ECG, one feature for age, and one feature for sex
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        # Get age, sex and root mean square of the leads.
        age, sex, ecg_features = get_features(header, recording, twelve_leads)
        #For øyeblikket 4 features 
        data[i, 0:4] = ecg_features
        data[i, 4] = age
        data[i, 4+1] = sex

        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1


    # Make cluster
    ohe = labels * 1
    my_cluster = []
    for i in range(len(ohe.T)):
        my_cluster.append(np.unique(np.where(ohe[np.where(ohe.T[i]==1)])[1]))
    
    # Train models.

    # Define parameters for random forest classifier.
    n_estimators = 3     # Number of trees in the forest.
    max_leaf_nodes = 100 # Maximum number of leaf nodes in each tree.
    random_state = 0     # Random state; set for reproducibility.

    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    leads = twelve_leads
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    #feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    #features = data[:, feature_indices]
    
    #Til nå kan features være lik data
    features = data

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    print("Making the 12-lead model")
    classifier = LabelSpacePartitioningClassifier(
        classifier = ClassifierChain(
            classifier= RandomForestClassifier(n_jobs=-1,n_estimators=n_estimators, verbose=1),
            require_dense = [False, True]
        ),
        require_dense = [True, True],
        clusterer = FixedLabelSpaceClusterer(clusters=my_cluster)
    )
    classifier.fit(features, labels)
    save_model(filename, classes, leads, imputer, classifier)

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    leads = six_leads
    filename = os.path.join(model_directory, six_lead_model_filename)

    #feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    #features = data[:, feature_indices]

    #Til nå kan features være lik data
    features = data[:,:8]

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    print("Making the 6-lead model")
    classifier = LabelSpacePartitioningClassifier(
        classifier = ClassifierChain(
            classifier= RandomForestClassifier(n_jobs=-1,n_estimators=n_estimators, verbose=1),
            require_dense = [False, True]
        ),
        require_dense = [True, True],
        clusterer = FixedLabelSpaceClusterer(clusters=my_cluster)
    )
    classifier.fit(features, labels)
    
    save_model(filename, classes, leads, imputer, classifier)

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    leads = three_leads
    filename = os.path.join(model_directory, three_lead_model_filename)

    #feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    #features = data[:, feature_indices]

    #Til nå kan features være lik data
    features = data[:,:6]

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    print("Making the 3-lead model")
    classifier = LabelSpacePartitioningClassifier(
        classifier = ClassifierChain(
            classifier= RandomForestClassifier(n_jobs=-1,n_estimators=n_estimators, verbose=1),
            require_dense = [False, True]
        ),
        require_dense = [True, True],
        clusterer = FixedLabelSpaceClusterer(clusters=my_cluster)
    )
    classifier.fit(features, labels)

    save_model(filename, classes, leads, imputer, classifier)

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    leads = two_leads
    filename = os.path.join(model_directory, two_lead_model_filename)

    #feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    #features = data[:, feature_indices]

    #Til nå kan features være lik data
    features = data[:,:6]

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    print("Making the 2-lead model")
    classifier = LabelSpacePartitioningClassifier(
        classifier = ClassifierChain(
            classifier= RandomForestClassifier(n_jobs=-1,n_estimators=n_estimators, verbose=1),
            require_dense = [False, True]
        ),
        require_dense = [True, True],
        clusterer = FixedLabelSpaceClusterer(clusters=my_cluster)
    )
    classifier.fit(features, labels)
    
    save_model(filename, classes, leads, imputer, classifier)

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    num_leads = len(leads)
    if num_leads == 2:
        data = np.zeros(6, dtype=np.float32)
    elif num_leads == 3:
        data = np.zeros(6, dtype=np.float32)
    elif num_leads == 6:
        data = np.zeros(8, dtype=np.float32)
    else:
        data = np.zeros(14, dtype=np.float32)
    age, sex, ecg_features = get_features(header, recording, leads)
    # 4 features for alle modeller for øyeblikket
    features_amount = 4
    data[0:features_amount] = ecg_features
    data[features_amount] = age
    data[features_amount+1] = sex

    # Impute missing data.
    features = data.reshape(1, -1)
    features = imputer.transform(features)

    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = labels.todense()
    labels = np.asarray(labels, dtype=np.int).ravel()
    #labels = np.asarray(labels, dtype=np.int)[0]

    #probabilities = classifier.predict_proba(features)
    #probabilities = probabilities.todense()
    probabilities = labels * 1.0
    probabilities = np.asarray(probabilities, dtype=np.float32).ravel()
    #probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    sample_freq = int(header.split()[2])
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]
    
    if num_leads == 2:
        # Lead II er første kolonne
        r_peaks = find_R_peaks(recording[0],sample_freq)
    else:
        r_peaks = find_R_peaks(recording[1],sample_freq)

    heartrate_r = (60/(np.diff(r_peaks)/sample_freq)) 
    heartrate_std = heartrate_r.std()
    heartrate_median = np.median(heartrate_r)
    # midlertidig løsning
    try:
        heartrate_min = heartrate_r.min()
    except:
        heartrate_min = 0
    # midlertidig løsning
    try:
        heartrate_max = heartrate_r.max()
    except:
        heartrate_max = 100
    ecg_features = np.array([heartrate_std,heartrate_median, heartrate_min, heartrate_max])
    return age, sex, ecg_features

