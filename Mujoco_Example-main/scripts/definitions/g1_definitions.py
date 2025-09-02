# class to hold the indices of the model instance
class Mujoco_IDX_12dof():
        
        # generalized positions
        POS_X = 0
        POS_Y = 1
        POS_Z = 2
        QUAT_W = 3
        QUAT_X = 4
        QUAT_Y = 5
        QUAT_Z = 6
        POS_LHP = 7
        POS_LHR = 8
        POS_LHY = 9
        POS_LKP = 10
        POS_LAP = 11
        POS_LAR = 12
        POS_RHP = 13
        POS_RHR = 14
        POS_RHY = 15
        POS_RKP = 16
        POS_RAP = 17
        POS_RAR = 18

        # generalized velocities
        VEL_X = 0
        VEL_Y = 1
        VEL_Z = 2
        ANG_X = 3
        ANG_Y = 4
        ANG_Z = 5
        VEL_LHP = 6
        VEL_LHR = 7
        VEL_LHY = 8
        VEL_LKP = 9
        VEL_LAP = 10
        VEL_LAR = 11
        VEL_RHP = 12
        VEL_RHR = 13
        VEL_RHY = 14
        VEL_RKP = 15
        VEL_RAP = 16
        VEL_RAR = 17

        # base index
        q_base_pos_idx = [POS_X, POS_Y, POS_Z]
        q_base_quat_idx = [QUAT_W, QUAT_X, QUAT_Y, QUAT_Z]
        v_base_vel_idx = [VEL_X, VEL_Y, VEL_Z]
        v_base_ang_idx = [ANG_X, ANG_Y, ANG_Z]

        # joints
        q_joint_idx = [POS_LHP, POS_LHR, POS_LHY, POS_LKP, POS_LAP, POS_LAR,
                       POS_RHP, POS_RHR, POS_RHY, POS_RKP, POS_RAP, POS_RAR]
        v_joint_idx = [VEL_LHP, VEL_LHR, VEL_LHY, VEL_LKP, VEL_LAP, VEL_LAR,
                       VEL_RHP, VEL_RHR, VEL_RHY, VEL_RKP, VEL_RAP, VEL_RAR]

        q_joint_idx = [POS_LHP, POS_LHR, POS_LHY, POS_LKP, POS_LAP, POS_LAR,
                       POS_RHP, POS_RHR, POS_RHY, POS_RKP, POS_RAP, POS_RAR]
        v_joint_idx = [VEL_LHP, VEL_LHR, VEL_LHY, VEL_LKP, VEL_LAP, VEL_LAR,
                       VEL_RHP, VEL_RHR, VEL_RHY, VEL_RKP, VEL_RAP, VEL_RAR]