import saki
import os

# Initialize the class
engine = saki.Saki()

template = \
    "version: 1 \
    settings: \
        neuron_cooldown: 1000000000.0; \
        refractory_time: 1; \
        max_connection_distance: 1; \
    hexagons:  \
        1,1,1;  \
        2,2,2;  \
        3,2,2;  \
    axons:  \
        1,1,1 -> 2,2,2;  \
    inputs:  \
        test_input: 1,1,1;  \
    outputs:  \
        test_output: 3,2,2;"

print("init")
uuid_str = engine.init_model(template)
print(f"UUID: {uuid_str}")

print("train")
training_inputs = {"test_input": [1.0, 2.5, 3.1]}
target_outputs = {"test_output": [0.1, 0.5, 0.9]}

for i in range(0, 1000):
    engine.train(uuid_str, training_inputs, target_outputs)

print("create and restore checkpoint")
file_path = f"/tmp/{uuid_str}"
engine.create_checkpoint(uuid_str, file_path)
engine.delete_model(uuid_str)
new_uuid_str = engine.init_model(template)
engine.restore_checkpoint(new_uuid_str, file_path)
os.remove(file_path)

print("request")
inputs = {"test_input": [1.0, 2.5, 3.1]}
outputs = {"test_output": [0.0, 0.0, 0.0]}

engine.request(new_uuid_str, inputs, outputs)
print(outputs)

engine.delete_model(new_uuid_str)
