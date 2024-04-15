from btgym.behavior_tree.base_nodes import Action
from btgym.behavior_tree import Status


class VHTAction(Action):
    can_be_expanded = True
    num_args = 1

    SurfacePlaces = {"floor","bathroomcabinet","bathroomcounter","towelrack","rug", "plate", "tvstand", "nightstand",  \
                     "kitchentable", "kitchencabinet", "kitchencounter", "fryingpan", "stove","oventray", "mat","tray",\
                      "bookshelf", "desk", "cabinet", "chair", "bench","sofa", "bed", "mousemat","radio", "boardgame",\
                     "couch","table","filing_cabinet","mousepad","bathtub"}  # put
    # 地板, 浴室柜, 浴室台面, 毛巾架, 地毯, 盘子, 电视柜, 床头柜, \
    # 厨房桌, 厨房橱柜, 厨房台面, 煎锅, 炉灶, 烤箱托盘, 门垫, 托盘, \
    # 书架, 书桌, 柜子, 椅子, 长凳, 沙发, 床, 鼠标垫, 收音机, 桌游, \
    # 沙发, 桌子, 文件柜, 鼠标垫, 浴缸

    SittablePlaces = {"Bathtub", "bed", "Bench", "chair", "rug", "sofa", "toilet"}  # sit
    # 浴缸, 床, 长凳, 椅子, 地毯, 沙发, 厕所


    CanOpenPlaces = {"bathroom_cabinet", "book", "bookshelf", "box", "coffee_pot", "cabinet", "closet", "clothes_pile",\
                     "coffeemaker", "cookingpot", "curtains", "desk", "dishwasher", "door", "folder", "fridge", \
                     "garbage_can", "hairproduct", "journal", "lotionbottle", "magazine", "microwave", "milk", \
                     "nightstand", "printer", "radio", "stove", "toilet", "toothpaste", "beer", "washing_machine",}  # open
    # 浴室柜, 书, 书架, 盒子, 咖啡壶, 柜子, 衣橱, 衣物堆, \
    # 咖啡机, 烹饪锅, 窗帘, 书桌, 洗碗机, 门, 文件夹, 冰箱, \
    # 垃圾桶, 护发产品, 日记本, 润肤乳瓶, 杂志, 微波炉, 牛奶,\
    # 床头柜, 打印机, 收音机, 炉灶, 厕所, 牙膏, 啤酒, 洗衣机

    CanPutInPlaces = {"bathroom_cabinet", "bookshelf", "box", "coffee_pot", "cabinet", "closet", "clothes_pile",\
                     "coffeemaker", "cookingpot", "desk", "dishwasher", "folder", "fridge", "garbage_can", "microwave", \
                     "nightstand", "printer", "stove", "toilet", "toothpaste", "washing_machine"}  # put in
    # 浴室柜, 书架, 盒子, 咖啡壶, 柜子, 衣橱, 衣物堆,
    # 咖啡机, 烹饪锅, 书桌, 洗碗机, 文件夹, 冰箱, 垃圾桶, 微波炉,
    # 床头柜, 打印机, 炉灶, 厕所, 牙膏, 洗衣机


    Objects = { "apple", "sportsball", "bananas", "barsoap", "bellpepper", "boardgame", "book", "box",
                "bread", "bread_slice", "broom", "bucket", "carrot", "cat", "cards", "cellphone", "chinesefood",
                "coffee_pot", "crayons", "chair", "candle", "chefknife", "chicken", "chocolatesyrup", "clock",
                "clothes_pants", "clothes_pile", "clothes_shirt", "condiment_bottle", "condiment_shaker",
                "cookingpot", "candybar", "crackers", "cereal", "creamybuns", "chips", "cucumber", "cupcake",
                "cutlery_knife", "cutlery_fork", "cutlets", "cutting_board", "dishwashingliquid", "dish_bowl",
                "plate", "dustpan", "facecream", "folder", "fryingpan", "globe", "glasses", "hairproduct", "hanger",
                "juice", "journal", "keyboard", "lemon", "lime", "lotionbottle", "kettle", "magazine", "milk",
                "milkshake", "mincedmeat", "mouse", "mug", "napkin", "notes", "orange", "pancake", "paper",
                "papertowel", "pear", "pen", "pie", "pillow", "plum", "potato", "poundcake", "pudding", "radio",
                "remote_control", "rug", "salad", "salmon", "slippers", "washing_sponge", "spoon", "sundae",
                "teddybear", "toy", "toiletpaper", "tomato", "toothbrush", "toothpaste", "towel", "towel_rack",
                "wallpictureframe", "wall_phone", "waterglass", "watermelon", "whippedcream", "wineglass",
                "alcohol", "beer", "wine"
               }
    # 苹果, 运动球, 香蕉, 条形肥皂, 灯笼椒, 桌游, 书, 盒子, 面包, 面包片, 扫帚, 桶, 胡萝卜,  猫, 卡片, 手机, 中餐,
    # 咖啡壶, 蜡笔, 椅子, 蜡烛, 厨师刀, 鸡肉, 巧克力糖浆, 时钟, 裤子, 衣服堆, 衬衫, 调味瓶, 调味瓶,
    # 烹饪锅, 糖果条, 饼干, 麦片, 奶油包, 薯片, 黄瓜, 纸杯蛋糕, 刀, 叉, 肉排, 砧板, 洗碗液, 碗,
    # 盘子, 簸箕, 面霜, 文件夹, 煎锅, 地球仪, 眼镜, 头发产品, 衣架, 果汁, 日记, 键盘, 柠檬, 酸橙, 乳液瓶, 水壶, 杂志, 牛奶,
    # 奶昔, 碎肉, 鼠标, 马克杯, 餐巾, 笔记, 橙子, 煎饼, 纸, 纸巾, 梨, 笔, 派, 枕头, 李子, 土豆, 磅蛋糕, 布丁, 收音机,
    # 遥控器, 地毯, 沙拉, 三文鱼, 拖鞋, 洗碗海绵, 勺子, 圣代, 泰迪熊, 玩具, 卫生纸, 番茄, 牙刷, 牙膏, 毛巾, 毛巾架,
    # 墙上相框, 墙电话, 玻璃水杯, 西瓜, 鲜奶油, 酒杯, 酒精, 啤酒, 葡萄酒


    # grab
    HasSwitchObjects = {"cellphone", "candle", "clock", "coffeemaker", "dishwasher", "fridge", "lightswitch", "kettle",\
                        "microwave", "pc", "printer", "radio", "remote_control", "stove", "tv", "toaster", "walltv", \
                        "wall_phone", "computer", "washing_machine"}
    # 手机, 蜡烛, 时钟, 咖啡机, 洗碗机, 冰箱, 开关, 水壶,
    # 微波炉, 电脑, 打印机, 收音机, 遥控器, 炉灶, 电视, 烤面包机, 墙壁电视,
    # 墙壁电话, 电脑, 洗衣机,

    # switch on #candle  cellphone wallphone washingmachine不行# faucet 浴室龙头
    AllObject = SurfacePlaces | SittablePlaces | Objects | CanOpenPlaces |\
                 CanPutInPlaces | HasSwitchObjects

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        pass

    def update(self) -> Status:
        # script = [f'<char0> [{self.__class__.__name__.lower()}] <{self.args[0].lower()}> (1)']

        if self.num_args == 1:
            script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1)']
        else:
            script = [
                f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1) <{self.args[1].lower()}> (1)']

        self.env.run_script(script, verbose=True, camera_mode="PERSON_FROM_BACK")  # FIRST_PERSON
        # print("script: ", script)
        self.change_condition_set()

        return Status.RUNNING
