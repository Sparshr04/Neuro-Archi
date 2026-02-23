% init_validation.m
% Initialization Script for Hybrid VTOL Benchmark Simulation
% Purpose: Define physics constraints, environment, and reference profiles.

clear; clc; close all;

%% 1. Physics Constraints (Strict)
% Vehicle Mass Properties
mass = 3.2; % kg
inertia_tensor = diag([0.08, 0.1, 0.15]); % kg*m^2 [Ixx, Iyy, Izz]

% Aerodynamics Reference
S_wing = 0.38; % m^2

% Actuator Dynamics
tau_actuator = 0.05; % Time constant (s) for first-order lag

%% 2. Simulation Parameters
dt = 0.0025; % Sample time (400 Hz)
T_final = 30.0; % Duration (s)

%% 3. Transition Profile Definition
% Time breakpoints for trajectory generation
% Phase 1: Hover/Climb (0-10s)
% Phase 2: Transition (10-20s) - Linear acceleration
% Phase 3: Cruise (20-30s)
time_waypoints = [0, 10, 20, 30]; 

% Velocity Command Profile (m/s)
% Ramp from 0 to 16 m/s during transition
velocity_waypoints = [0, 0, 16, 16]; 

% Altitude Command Profile (m)
% Climb to 30m in Phase 1, maintain hold
altitude_waypoints = [0, 30, 30, 30];

% Create Timeseries for Simulink (Linear Interpolation)
ref_time = 0:dt:T_final;
ref_velocity = interp1(time_waypoints, velocity_waypoints, ref_time, 'linear');
ref_altitude = interp1(time_waypoints, altitude_waypoints, ref_time, 'linear');

% Pack into structure or timeseries objects if needed by specific blocks
% For basic Lookup Tables, the array variables are sufficient.
disp('Initialization Complete: Parameters Loaded.');
