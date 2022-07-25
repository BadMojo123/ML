import MieszkoI


# !!!!!!!!!!!!!!initilalizer

testSet = 0.9
l1_size=[800]
l2_size=[800]
l3_size=[800]
l4_size=0
l5_size=0
l1_activation = ['selu']
l2_activation = ['selu']
l3_activation='selu'
l4_activation='selu'
l5_activation='selu'
epochs=100
validation_split=0.01
batch_size=32
# l1_activation = ['softmax','tanh','selu','relu']
loss= ['categorical_hinge']
# loss= ['categorical_hinge','binary_crossentropy', 'mean_squared_error','squared_hinge','logcosh']
# optimizer=['adam','rmsprop','SGD','Nadam','TFOptimizer']
optimizer=['Nadam']
metrics=['accuracy']

for l1a in l1_activation:
	for l2a in l2_activation:
		for li in loss:
			for oi in optimizer:
				for l1s in l1_size:
					for l2s in l2_size:
						for l3s in l3_size:
							MieszkoI.MieszkoI_learn(
								testSet = testSet,
								l1_size=l1s,
								l2_size=l2s,
								l3_size=l3s,
								l4_size=l4_size,
								l5_size=l5_size,
								l1_activation=l1a,
								l2_activation=l2a,
								l3_activation=l3_activation,
								l4_activation=l4_activation,
								l5_activation=l5_activation,
								loss=li,
								optimizer=oi,
								metrics=metrics,
								epochs=epochs,
								validation_split=validation_split,
								batch_size=batch_size,
								modelName="klony/Mieszko1527196947.7612674.h5"
								)
