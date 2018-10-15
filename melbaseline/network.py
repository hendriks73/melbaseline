from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv1D, Dropout, AveragePooling1D, BatchNormalization, \
    GlobalAveragePooling1D, Activation


def create_model(input_shape=(40, 9), output_dim=200, filters=128, dropout=0.1):
    print('Dropout: {}'.format(dropout))
    print('Filters/layer: {}'.format(filters))
    filter_size = 3
    pool_size = 2

    input = Input(shape=input_shape)
    x = Conv1D(filters, filter_size, padding='same', activation='elu', name="Conv0")(input)
    if dropout > 0.:
        x = Dropout(dropout, name='DO0')(x)
    x = AveragePooling1D(pool_size, name="AP0")(x)
    x = BatchNormalization(name="BN0")(x)

    x = Conv1D(filters*2, filter_size, padding='same', activation='elu', name="Conv1")(x)
    if dropout > 0.:
        x = Dropout(dropout, name='DO1')(x)
    x = AveragePooling1D(pool_size, name="AP1")(x)
    x = BatchNormalization(name="BN1")(x)

    x = Conv1D(filters*4, filter_size, padding='same', activation='elu', name="Conv2")(x)
    if dropout > 0.:
        x = Dropout(dropout, name='DO2')(x)
    x = AveragePooling1D(pool_size, name="AP2")(x)
    x = BatchNormalization(name="BN2")(x)

    x = Conv1D(filters*8, filter_size, padding='same', activation='elu', name="Conv3")(x)
    if dropout > 0.:
        x = Dropout(dropout, name='DO3')(x)
    x = BatchNormalization(name="BN3")(x)

    x = Conv1D(output_dim, 1, padding='same', activation='elu', name="DimReduction")(x)
    x = BatchNormalization(name="BN")(x)
    x = GlobalAveragePooling1D()(x)
    x = Activation(activation='sigmoid')(x)

    return Model(inputs=input, outputs=x)