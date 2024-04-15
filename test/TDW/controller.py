from transport_challenge import Transport

m = Transport(debug=True)
# Initializes the scene.
status = m.init_scene(scene="2a", layout=1)
print(status)  # ActionStatus.success

# Prints a list of all container IDs.
print(m.containers)
# Prints a list of all target object IDs.
print(m.target_objects)

m.end()