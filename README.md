# Flyappy Autonomy Test Game

This repository contains the coding test for the Flyability Autonomy team.

## Game Description

Flyappy is in trouble again! This time it went into space and landed in an asteroid
belt. If Flyappy collides with the asteroids it would be fatal. Luckily Flyappy remembered
his laser scanner that provides distance measurements. It will give you its velocity and
the laserscans in return for an acceleration input. Flyappy only asks for 60 seconds of
your guidance. Help Flyappy go through as many asteroid lines as possible before the time
runs out!

[![Flyappy: Score 106](flyappy_cover.png)](106.mp4)

A list of score reached in 20 consecutive runs (**High Score:** 110 and **Average Score**: 98.6):
- Run 1: 110
- Run 2: 99
- Run 3: 106
- Run 4: 103
- Run 5: 100
- Run 6: 106
- Run 7: 108
- Run 8: 102
- Run 9: 107
- Run 10: 58
- Run 11: 105
- Run 12: 100
- Run 13: 71
- Run 14: 108
- Run 15: 104
- Run 16: 102
- Run 17: 70
- Run 18: 103
- Run 19: 102
- Run 20: 108

## A few words about the solution
The version that you see in the video uses a model predictive controller for reference tracking. For such tasks, it is convenient to write the MPC problem in delta formulation. I formulate the QP problem such that the dynamics are substituted.

The gate detection computes two possible gaps by first sorting the pointcloud according to the y-component and then by measuring the difference between two consecutives points. The algorithms considers a gap as big enough if it is at least twice the size of the bird. If the second biggest gap satisfies this condition then this gap is chosen. The gate position is then computed based on the average of the upper and lower point that represent the gap.

There are in total four states in which the bird can be:
- **INIT**: During this state the bird measures the upper and lower fence. This should help then to filter out points registered on the ground or ceiling.
- **MOVE_FORWARD**: Here, the bird simply moves forward until it reaches 4.7 m, which is close engough to compute the gate position.
- **TUNNEL**: This is a constrained state, where the bird's y-component is kept fix.
- **TARGET**: As the name suggest, this is the phase where the bird adjusts it height. If needed (due to big height difference: at least 1.0 m) it will adjust the x-speed.

Initially, I started by implementing a PID controller. The problem there was the oscillation and also it was not able to handle high speeds. So, I switch to a linear quadratic regulator. With that I could already reach scores of at least 40. But what was missing, was the possibility to add constraints, especially on the control input. That's why ended up using a model prective controller. For the tuning of the LQR and the MPC I implemented a few matlab scripts, mainly because it is easier to formulate MPC problems there and to use it as verification for the C++ implementation of the LQR and MPC. You can find those here: [Matlab Files](https://github.com/riccardofeingold/MPC_LQR_tuning_for_flyappy_autonomy_game)

## Getting Started
### Setup environment
First, install the qpOASES cloning their repo into the src folder:
```bash
git clone https://github.com/coin-or/qpOASES.git
```
Then go into the qpOASES folder and run *make*:
```bash
make
```
Then install OpenCV:
```bash
sudo apt-get install ros-noetic-vision-opencv
```

After that you can build the actual ros packages using:
```bash
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

After compilation, at the root of the workspace, source the workspace in the terminal
you want to run the game:
```
source devel/setup.bash
```

Run the C++ version:
```
roslaunch flyappy_autonomy_code flyappy_autonomy_code_cpp.launch
```

### Setup the running environment

*This game has been tested with Ubuntu 20.04 running ROS Noetic and Python 3.8.10.*

There are two recommended options for running the game. Either download the VirtualBox
image that comes with a complete Ubuntu 20.04 setup or add the necessary packages to
your system to compile and run the game.

#### Option 1 - Using VirtualBox

First install VirtualBox on your system
[VirtualBox wiki link](https://www.virtualbox.org/wiki/Downloads).

Then download the Ubuntu 20.04 image that we have preconfigured with ROS and the
necessary packages
[Image link](https://drive.google.com/file/d/14XXwDKfOUH8pCh18CagoJmq6qFxY289Q/view?usp=share_link).

Once downloaded add the image to your VirtualBox and boot up Ubuntu. The username and
password are both **flyatest**.

Note: You might have issues and latencies using VirtualBox, depending on your system and
your OS.

#### Option 2 - Using your system

If you already have Ubuntu 20.04 on your system, great. If not, you can either install
it on your machine (dual-boot or full installation) or boot from an USB flash drive.
You can follow
[Ubuntu tutorial: Install Ubuntu desktop](https://ubuntu.com/tutorials/install-ubuntu-desktop).
If your system is running Windows 11, you can maybe try
[Ubuntu tutorial: Install Ubuntu on WSL2 on Windows 11 with GUI support](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support)
(not tested).

Make sure ROS Noetic is installed by following the
[ROS install guide](http://wiki.ros.org/noetic/Installation/Ubuntu).

Make sure Pygame for Python3 is installed:
```
sudo apt install python3-pygame
```

### Setup the workspace

Open a terminal and run:
```
mkdir flyappy_ws
cd flyappy_ws
```

Clone the repository in the source of the workspace:
```
git clone https://github.com/Flyability/flyappy_autonomy_test_public.git src/flyappy_autonomy_test_public
```

### Compilation

At the root of the workspace, run the catkin_make command:
```
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

### Run The Game

After compilation, at the root of the workspace, source the workspace in the terminal
you want to run the game:
```
source devel/setup.bash
```

There are two ROS launch files depending on if you want to run the game with C++ or
Python autonomy code (whatever you prefer).

For Python, run:
```
roslaunch flyappy_autonomy_code flyappy_autonomy_code_py.launch
```

For C++, run:
```
roslaunch flyappy_autonomy_code flyappy_autonomy_code_cpp.launch
```

A GUI will become visible with the game start screen. For now the autonomy code does not
do anything other than printing out some laser ray and end game information. To start
the game, press any arrow key.

You can then add velocity in a wanted direction by pressing the arrow keys
&larr;&uarr;&darr;&rarr;.

Notice that it is not possible to go backwards and if Flyappy hits an obstacle the game
stops.

## Automate Flyappy

Now that we have gotten familiar with the game, we want to control Flyappy autonomously.
To do this, a Python and a C++ template have been provided.

### Modifying the code

The templates are located in the **flyappy_autonomy_code** folder. Be aware that you
are not meant to change the files in **flyappy_main_game**.

For using python, modify the file **flyappy_autonomy_code_node.py** in the **scripts**
folder, and add any Python files if needed.

For using C++, modify (or add) any files in the **include**, **src**, **tests** folders
and, if needed, the **CMakeLists.txt**.

Take your pick.

To get the state of Flyappy, its velocity and laserscans data are published on 2 ROS
topics. An acceleration command can be given on a ROS topic for actuating Flyappy.

The callbacks and publisher are provided in the code.

### Handing In

To hand in your game solution, please send your **flyappy_autonomy_test_public**
repository in a ZIP file by email.

### Other Information

I hope you will have fun solving this little game. If you have any questions or need
other game information either write us or look around in the **flyappy_main_game**
folder. Here is some other helpful information for solving the task.

* Scaling: 1 pixel = 0.01 meter
* Game and sensor update rates: 30 fps
* The velocity measurement is noise free
* Max acceleration x: 3.0 m/s^2
* Max acceleration y: 35.0 m/s^2
* Axis convention: x &rarr;, y &uarr;
* [LaserScan message definition](http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html)

| Value        | Unit           | Topic  |
| ------------- |:-------------:| :-----:|
| Velocity      | m/s           | /flyappy_vel |
| Acceleration  | m/s^2         | /flyappy_acc |
| LaserScan     | Radians, meters      | /flyappy_laser_scan |
| GameEnded     | No unit      | /flyappy_game_ended |
