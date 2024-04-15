from magnebot import MagnebotController

c = MagnebotController()
c.init_floorplan_scene(scene="1a", layout=0, room=0)
x = 30
z = 16
print(c.occupancy_map[x][z]) # 0 (free and navigable position)
print(c.get_occupancy_position(x, z)) # (1.1157886505126946, 2.2528389358520506)
c.end()