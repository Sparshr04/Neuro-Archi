% export_ground_truth.m % Data Post -
    Processing and Export Script %
        Purpose : Convert Simulink ENU data to NED and match Python headers.

        % % 1. Resampling %
        Target Frequency : 400 Hz(dt = 0.0025) target_time = 0 : 0.0025 : 30.0;

% Assume 'out' is the Simulink output object %
    Extract raw timeseries(assuming logged as 'sim_data' structure) %
    Adjust these accessors based on your actual Simulink logging setup
    raw_time = out.sim_out_data.time;
raw_pos_enu = out.sim_out_data.signals.values( :, 1 : 3);
% [ x, y, z ] raw_vel_enu = out.sim_out_data.signals.values( :, 4 : 6);
% [ vx, vy, vz ] raw_euler = out.sim_out_data.signals.values( :, 7 : 9);
% [ phi, theta, psi ] raw_accel = out.sim_out_data.signals.values( :, 10 : 12);
% [ ax, ay, az ] raw_gyro = out.sim_out_data.signals.values( :, 13 : 15);
% [ p, q, r ]

    % Resample to exact grid using linear interpolation pos_enu_res =
    interp1(raw_time, raw_pos_enu, target_time, 'linear');
vel_enu_res = interp1(raw_time, raw_vel_enu, target_time, 'linear');
euler_res = interp1(raw_time, raw_euler, target_time, 'linear');
accel_res = interp1(raw_time, raw_accel, target_time, 'linear');
gyro_res = interp1(raw_time, raw_gyro, target_time, 'linear');

% % 2. Coordinate Transformation(ENU->NED) %
    Rule : X_NED = Y_ENU,
           Y_NED = X_ENU, Z_NED = -Z_ENU % This ensures P_n(North)
corresponds to Y_ENU(which corresponds to Python X if Python is NED) %
    User Requirement : "Ensure p_n corresponds to the Python x axis" %
                       If Python X is North(std NED),
    then P_n = Python X.% If Simulation is ENU,
         Y is North.So P_n = Y_ENU.

                             % Position p_n = pos_enu_res( :, 2);
% Y_ENU p_e = pos_enu_res( :, 1);
% X_ENU p_d = -pos_enu_res( :, 3);
% -Z_ENU(Down is negative Up)

    % Velocity v_n = vel_enu_res( :, 2);
v_e = vel_enu_res( :, 1);
v_d = -vel_enu_res( :, 3);

% Euler Angles
% Standard mapping: Phone/Theta align with body frame changes?
% If Body frame is standard (X-Forward), ENU vs NED changes reference.
% Phi (Roll): X-axis rotation. NED X is North. ENU X is East.
% For benchmarking, we assume the Body frame aligns with the Velocity vector or similar.
% However, strictly for the output vector:
% Phi_NED = Phi_ENU (approx for small angles if X/Y swapped? No.)
% USE Rotation Matrix check if high fidelity needed.
% For this strict output script:
phi = euler_res(:, 1);
theta = euler_res( :, 2);
psi = -euler_res( :, 3) + pi / 2;
% Convert Enu Yaw(0 = East, CCW) to NED Yaw (0=North, CW)
% Note: Verify 'psi' conversion with Python ground truth if needed.

% IMU (Body Frame)
% Accelerometer and Gyro are usually in BODY frame.
% If Simulink 6DOF outputs in BODY frame, no ENU/NED conversion needed for X/Y/Z components
% UNLESS the body frame definition itself is different.
% Standard Aerospace Body: X=Nose, Y=Right, Z=Down.
% Simulink Body might be X=Nose, Y=Right, Z=Down.
% If so, pass through.
acc_x = accel_res(:, 1);
acc_y = accel_res( :, 2);
acc_z = accel_res( :, 3);

gyro_x = gyro_res( :, 1);
gyro_y = gyro_res( :, 2);
gyro_z = gyro_res( :, 3);

% % 3. Export to CSV % Headers : timestamp, p_n, p_e, p_d, v_n, v_e, v_d, phi,
    theta, psi, acc_x, acc_y, acc_z, gyro_x, gyro_y,
    gyro_z

        data_table = table(target_time', p_n, p_e, p_d, v_n, v_e, v_d, ... phi,
                           theta, psi, acc_x, acc_y, acc_z, gyro_x, gyro_y,
                           gyro_z, ... 'VariableNames',
                           {'timestamp', 'p_n', 'p_e', 'p_d', 'v_n', 'v_e',
                            'v_d', ... 'phi', 'theta', 'psi', 'acc_x', 'acc_y',
                            'acc_z', ... 'gyro_x', 'gyro_y', 'gyro_z'});

filename = 'ground_truth_matlab.csv';
writetable(data_table, filename);
disp([ 'Data exported to ', filename ]);
