# Self Driving Car Control Design

## Self Driving Cars Longitudinal and Lateral Control Design 

In this project, I implemented a  controller in Python and use it to drive a car autonomously around a track in Carla Simulator.

### Project notes:

The waypoints include positions as well as the speed the vehicle should attain. As a result, the waypoints
become the reference signal for our controller and navigating to all the waypoints effectively completes the full track.


Since the controller reference contains both position and speed, we need to implement both __Longitudinal and Lateral Control__.


The output of the controller will be the vehicle __throttle__, __brake__ and __steering angle__ commands.

The throttle and brake come from the Longitudinal speed control and the steering comes from our Lateral Control.


If you want to make your own controller design,  I highly recommend that you should consider the following points:

If you open the simulator directory and navigate to the __Course1FinalProject folder__, you will see a file named __controller2d.py__

### These controller functions are as the following:

![Screenshot_6](https://user-images.githubusercontent.com/30608533/55581623-fb1e8a80-5725-11e9-9764-8145816374df.jpg)


You can adjust and re-create these functions according to your own custom controller design.



###  The Race Tack including waypoints:

![Screenshot_4](https://user-images.githubusercontent.com/30608533/55582000-0e7e2580-5727-11e9-83dd-4bafa30da590.jpg)
