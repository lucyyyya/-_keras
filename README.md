实训总结
这次实训的主题有两个，看了最终的答辩发现大家都选择了第二个主题，以传播为主题围绕这次新冠病毒来制作一个游戏。然后大家选择的游戏方向也都各不相同，比如fps、经营类、塔防等等。我们组也是塔防，为了迎合主题我们前期花了一周的时间进行讨论，然后程序组一开始也是做了基础的塔防功能，我们从生成敌人，地图构建以及塔的基本结构开始，等到策划将详细的游戏机制写出来之后再进行完善。

负责工作及完成情况：
可能会和创新点重复，也可以直接先看创新点
因为程序组并没有进行很确切的分工，有一个组员首先开始构建了敌人的行走路径，然后我就去建造了塔的管理类，接着剩下的两个组员也加入了进来，很神奇的是也没有产生冲突，一个负责了塔的攻击功能，一个负责敌人的血量及特效的添加等。
我主要是写了两个管理类和两个数据类
  
  
GameManager
这是用来管理游戏结束界面的。
界面很简单就是一个画布加一个Text和几个按钮，然后通过animation状态机选择触发对应的动画
 
 
重玩和菜单以及下一关的按钮其实就场景重新加载的语句
 
TowerManager
这个类负责的东西有些冗杂，因为当时只是用来负责塔的数据接收以及实例化，后面发现所有可以从Insepector面板获取的数据都需要通过这个类来传输，比如TowerData里面就是塔的数据存储，他的数据获取是通过manager的两个公共属性数据从面板获取值然后传递给建塔时塔的绑定，
 
还有ATP（就是金钱）的消耗也被我写在了这里，这个其实可以单独写成一个类，不过因为占的篇幅不多，当时想着建塔要消耗金钱就直接在画布上加了个text，后面又给他添加了动画什么的，再去看的时候发现占的篇幅还不小。

这里是金钱变动函数的处理
 
对要创建的塔的数值的描述
 
塔的选择是通过toggle实现的，当时为了避免塔种类过多，加了个对toggle的显示和隐藏的函数
 
当选中了其中某个toggle时，变将对应的towerdata传给一个私有的towerdata类
 
然后将对应的towerdata传过去，并更新面板的描述，然后将toggle放大表示被选中。
 
主要讲一下建塔时的一些相关检测。选择了射线碰撞了检测塔的建立位置，并且给cube设置了TowerCube层，防止塔建立到cube以外的prefab上。并且做出了判定，满足三个要求才能进行下一步的逻辑判断，比如ATP（资源）是否充足等
 
作为塔的管理类，有很多面板上的功能按键的实现也放在了这里，当然具体的逻辑实现在别的脚本中，这里主要做数据传输用，以及一些面板的隐藏和显示。
 
其中主要是利用了toggle和button在Inspector里面的自带功能来实现点击的数据传输
  

同时因为对塔的描述的canves不是实时都显现的，于是给其增加了隐藏和显示的功能 主要是通过SetActive函数，后面很多相关的功能也是通过这个实现的。
TowerData
 
通过将其保存在一个类中来实现管理，每创建一个新的塔的时候，都通过TowerData来获取他的初始数据，并在点击塔的功能按键时传输相应的Update数据
TowerCube
分化和分裂的具体实现都在这个类里面
 
因为塔的建立都和TowerCube密切关联，从一开始选中cube的时候，到怪物感染cube影响塔的生存等，都有一定的影响，所以将塔的具体功能实现放在了这里，也带来一定的便利
 
因为塔和cube的关联是从buildtower开始的，所以我们也将数据的互通放在了这个函数里，并设置了一个towerdata私有数据方便取值。
 

然后在摧毁的时候将cube上关于塔的数据全部清零
 

创新点：
1、
因为主要负责了塔数据的维护，塔的具体攻击行为是另一个同学负责的，他只从manager中获取数据，因此manager和towercube类中都维护了一个私有的towerdata数据，
manager中随时更新当前被选中的塔的原始数据
 
用到的地方也很多，比如建塔的时候要用这个当前被选中的属性来建立，然后还要传给这个塔绑定的prefab的攻击脚本里

 
 
towercube维护自身方格上放置的塔的数据
 
在build的时候才会被初始化
 
主要是分裂和分化的时候用来更改数值
 
2、
既然有被选中的塔数据，为了实现塔的升级面板，manager还维护了一个当前被选中的塔数据
 
这里可能会比较疑惑为什么是cube类，cube不是放置塔的方块吗，这里上面也说了，cube类还维护data类型，在build函数生效的时候便会同步，也就是这个时候讲放置在这个方块上的塔数据传了进来。
 
 
然后用更新后的数据来替换面板
 
3、
同时因为每个塔的种类不一样，在分裂次数上也不相同（可以理解为升级），所以在towerdata中我们添加了一个枚举类型TowerType，然后给每个类都加一个type类型以选择他是什么塔。
 
实训感想：
这次的实训让我更熟练的学会了如何使用码云将github中的代码同步以及上传，github的上传速度在某一周让我绝望，于是上网搜索了一下发现还可以用码云，瞬间体会到了飞一般的感觉。
咳现在讲正题。这次的团队分工虽然没有组内开会讨论过，但是我们建了一个程序组的群进行交流，很神奇的是也没有产生过分歧，并且大家也十分默契的将某一个人没做的东西自己说明了一下之后便开始动工了，策划出来之后要添加新的功能时，敌人的放给了负责敌人的同学，塔的分裂分化自然是我，然后塔攻击的同学负责了方块的感染，后面有些遗漏的东西我们便在群上说还有xxx没弄谁来搞一下，也很快就有人回应并快速的解决了。可能有些遗憾的就是没有开过组内的小会吧，不过大会肯定是开过的啦，最后美术组的场景及特效的添加也非常的棒。总体来说我觉得这次实训的收获还是很多的。


