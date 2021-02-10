from numpy import int64

from randomForest_classifier.predict import make_prediction
from randomForest_classifier.processing.data_management import load_dataset


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='test.csv')
    single_test_input = test_data[0:1]

    # When
    subject = make_prediction(input_data=single_test_input)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], int64)
    assert subject.get('predictions')[0] == 0


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name='test.csv')
    original_data_length = len(test_data)
    multiple_test_input = test_data

    # When
    subject = make_prediction(input_data=multiple_test_input)
    print('LEN',subject.get('predictions'))

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 2421

    # We expect some rows to be filtered out
    assert len(subject.get('predictions')) != original_data_length
