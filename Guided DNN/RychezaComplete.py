import MieszkoI

testSet = 0.9
l1_size = 700
l2_size = 1400
l3_size = 700
l4_size = 700
l5_size = 0
l1_activation = 'selu'
l2_activation = 'selu'
l3_activation = 'selu'
l4_activation = 'selu'
l5_activation = 'selu'
epochs = 100
validation_split = 0.01
batch_size = 64
# l1_activation = ['softmax','tanh','selu','relu']
loss = 'categorical_hinge'
# loss= ['categorical_hinge','binary_crossentropy', 'mean_squared_error','squared_hinge','logcosh']
# optimizer=['adam','rmsprop','SGD','Nadam','TFOptimizer']
optimizer = 'Nadam'
metrics = 'accuracy'

MieszkoI.MieszkoI_learn(
    testSet=testSet,
    l1_size=l1_size,
    l2_size=l2_size,
    l3_size=l3_size,
    l4_size=l4_size,
    l5_size=l5_size,
    l1_activation=l1_activation,
    l2_activation=l2_activation,
    l3_activation=l3_activation,
    l4_activation=l4_activation,
    l5_activation=l5_activation,
    loss=loss,
    optimizer=optimizer,
    metrics=metrics,
    epochs=epochs,
    validation_split=validation_split,
    batch_size=batch_size
)
