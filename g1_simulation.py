#!/usr/bin/env python

import mujoco
import glfw

import numpy as np
import yaml
import time

from definitions.g1_definitions import Mujoco_IDX_12dof


# main function to run the system
if __name__ == '__main__':
    
    # load the config file
    config_file = './config/g1_config.yaml'
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)   

    # load the mujoco model 
    mj_model_path = config['MODEL']['xml_path']
    model = mujoco.MjModel.from_xml_path(mj_model_path)
    data = mujoco.MjData(model)

    # setup the glfw window
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    window = glfw.create_window(1920, 1080, "Robot", None, None)
    glfw.make_context_current(window)

    # set the window to be resizable
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)

    # create camera to render the scene
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    cam.distance = 2.0
    cam.elevation = -15
    cam.azimuth = 135

    # turn on reaction forces
    if config['VISUALIZATION']['grf']:
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # create the scene and context
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_200)

    # create the index objects
    mj_idx = Mujoco_IDX_12dof()

    # gains
    kp = np.array(config['GAINS']['kps'])
    kd = np.array(config['GAINS']['kds'])

    # default joint positions
    q0_joint = np.array(config['INITIAL_STATE']['default_angles'])
    v0_joint = np.zeros(len(q0_joint))

    # set the intitial state
    data.qpos = np.zeros(model.nq)
    data.qvel = np.zeros(model.nv)
    data.qpos[mj_idx.q_base_pos_idx[0]] = config['INITIAL_STATE']['px']
    data.qpos[mj_idx.q_base_pos_idx[1]] = config['INITIAL_STATE']['py']
    data.qpos[mj_idx.q_base_pos_idx[2]] = config['INITIAL_STATE']['pz']
    data.qpos[mj_idx.q_base_quat_idx[0]] = 1.0
    data.qpos[mj_idx.q_joint_idx] = q0_joint
    data.qvel[mj_idx.v_joint_idx] = v0_joint

    # timers
    t_sim = 0.0
    dt_sim = model.opt.timestep
    max_sim_time = config['SIM']['t_max']

    # for rendering frequency
    hz_render = config['SIM']['hz_render']
    dt_render = 1.0 / hz_render
    counter = 0
    t1_wall = 0.0
    t2_wall = 0.0
    dt_wall  = 0.0

    # for control frequency
    hz_control = config['SIM']['hz_control']
    dt_control = 1.0 / hz_control
    decimation = int(dt_control / dt_sim)

    # for timing the total sim time
    t0_real_time = time.time()

    # main simulation loop
    while (not glfw.window_should_close(window)) and (t_sim < max_sim_time):

        # get the current sim time
        t0_sim = data.time

        # advance the simulation time until time to render
        while (t_sim - t0_sim) < (dt_render):

            # start inner loop wall clock
            t1_wall = time.time()

            # get current sim time and state
            t_sim = data.time

            # do control at a desired rate
            if counter % decimation == 0:

                # extract the joint states
                q = data.qpos
                v = data.qvel

                # get the joint positions and velocities
                q_joints = q[mj_idx.q_joint_idx]
                v_joints = v[mj_idx.v_joint_idx]

                # TODO: do contorl here, you can only control the joints
                q_joints_des = np.zeros_like(q_joints)
                v_joints_des = np.zeros_like(v_joints)

                # Example: set desired joint angles to zero
                q_joints_des[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t_sim)

                # compute torque from PID
                u = -kp * (q_joints - q_joints_des) - kd * (v_joints - v_joints_des)

            # set the control torque
            data.ctrl[:] = u

            # advance the simualtion 
            mujoco.mj_step(model, data)

            # wait until next control update
            t2_wall = time.time()
            dt_wall = t2_wall - t1_wall
            dt_sleep = dt_sim - dt_wall
            if dt_sleep > 0.0:
                time.sleep(dt_sleep)

        # update the camera position
        px_base = data.qpos[mj_idx.q_base_pos_idx[0]]
        py_base = data.qpos[mj_idx.q_base_pos_idx[1]]
        pz_base = data.qpos[mj_idx.q_base_pos_idx[2]]
        cam.lookat[0] = px_base
        cam.lookat[1] = py_base
        cam.lookat[2] = pz_base

        # update the scene
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

        # render the scene
        mujoco.mjr_render(viewport, scene, context)

        # display simulation time overlay
        t1_real_time = time.time()
        label_text = f"Sim Time: {t_sim:.2f} sec \nWall Time: {t1_real_time - t0_real_time:.3f} sec"
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_200,   # font scale
            mujoco.mjtGridPos.mjGRID_TOPLEFT,      # position on screen
            viewport,                              # this must be the MjrRect, not context
            label_text,                            # main overlay text (string, not bytes)
            "",                                    # optional secondary text
            context                                # render context
        )

        # swap the buffers
        glfw.swap_buffers(window)

        # poll for window events
        glfw.poll_events()

    print(f"Total simulation time: {t1_real_time - t0_real_time:.3f} seconds" )