#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

using namespace casadi;

int main (int argc, char **argv)
{
    ros::init(argc, argv, "hello");
    ros::NodeHandle nh;
    ROS_INFO("Node has been started");

    std::string FUNCTIONS_DIR;
    nh.getParam("admm_test/functions_path", FUNCTIONS_DIR);

    ros::Rate rate(1);

    casadi::Function ocpX_function = casadi::Function::load(FUNCTIONS_DIR + "/ocpX.casadi");
    ROS_INFO_STREAM("ocpX function: " << ocpX_function);

    casadi::Function ocpZ_function = casadi::Function::load(FUNCTIONS_DIR + "/ocpZ.casadi");
    ROS_INFO_STREAM("ocpZ function: " << ocpZ_function);

    casadi::Function sim_asv_dyn_function = casadi::Function::load(FUNCTIONS_DIR + "/sim_asv_dyn.casadi");
    ROS_INFO_STREAM("sim_asv_dyn function: " << sim_asv_dyn_function);

    int nx = ocpX_function.size1_in(2);
    ROS_INFO_STREAM("nx: " << nx);

    int Nhor_plus_1 = ocpX_function.size2_in(2);
    int Nhor = Nhor_plus_1 - 1;
    ROS_INFO_STREAM("Nhor: " << Nhor);

    int number_of_robots = ocpX_function.size1_in(4)/nx;
    ROS_INFO_STREAM("number_of_robots: " << number_of_robots);


    //---------------------------------------------------------------
    // setting up input and output matrices
    Eigen::MatrixXd xref(Eigen::MatrixXd::Zero(1, 1));
    Eigen::MatrixXd yref(Eigen::MatrixXd::Zero(1, 1));
    Eigen::MatrixXd li(Eigen::MatrixXd::Zero(nx, Nhor_plus_1));
    Eigen::MatrixXd ci(Eigen::MatrixXd::Zero(nx, Nhor_plus_1));
    Eigen::MatrixXd lji(Eigen::MatrixXd::Zero(nx*number_of_robots, Nhor_plus_1));
    Eigen::MatrixXd cji(Eigen::MatrixXd::Zero(nx*number_of_robots, Nhor_plus_1));
    Eigen::MatrixXd X_0(Eigen::MatrixXd::Zero(nx, 1));

    Eigen::MatrixXd x_res(1, Nhor_plus_1);
    Eigen::MatrixXd y_res(1, Nhor_plus_1);
    Eigen::MatrixXd u_res(1, Nhor);
    Eigen::MatrixXd v_res(1, Nhor);

    // casadi::DM xref(1, 1);
    // casadi::DM yref(1, 1);
    // casadi::DM li(nx, Nhor_plus_1);
    // casadi::DM ci(nx, Nhor_plus_1);
    // casadi::DM lji(nx*number_of_robots, Nhor_plus_1);
    // casadi::DM cji(nx*number_of_robots, Nhor_plus_1);
    // casadi::DM X_0(nx, 1);

    // casadi::DM x_res(1, Nhor_plus_1);
    // casadi::DM y_res(1, Nhor_plus_1);
    // casadi::DM u_res(1, Nhor);
    // casadi::DM v_res(1, Nhor);

    xref(0,0) = 1.0;
    yref(0,0) = 1.0;

    //----------------------------------------------------------------
    // Option 1: Setting up memory(-overhead-)less evaluation for ocpX
    std::vector<const double*> arg_ocpX(ocpX_function.sz_arg());
    std::vector<double*> res_ocpX(ocpX_function.sz_res());
    std::vector<casadi_int> iw_ocpX(ocpX_function.sz_iw());
    std::vector<double> w_ocpX(ocpX_function.sz_w());

    int mem_ocpX = ocpX_function.checkout();

    arg_ocpX[0] = &xref(0,0);
    arg_ocpX[1] = &yref(0,0);
    arg_ocpX[2] = &li(0,0);
    arg_ocpX[3] = &ci(0,0);
    arg_ocpX[4] = &lji(0,0);
    arg_ocpX[5] = &cji(0,0);
    arg_ocpX[6] = &X_0(0,0);


    res_ocpX[0] = &x_res(0,0);
    res_ocpX[1] = &y_res(0,0);
    res_ocpX[2] = &u_res(0,0);
    res_ocpX[3] = &v_res(0,0);    


   
    while (ros::ok()) {
        // Option 1: Memory(-overhead-)less evaluation of ocpX
        ocpX_function(casadi::get_ptr(arg_ocpX), casadi::get_ptr(res_ocpX), casadi::get_ptr(iw_ocpX), casadi::get_ptr(w_ocpX), mem_ocpX);

        ROS_INFO_STREAM("Hello: " << u_res);

        rate.sleep();
    }

    ocpX_function.release(mem_ocpX);
}


    // //---------------------------------------------------------------
    // // Option 2: Setting up normal evaluation for ocpX
    // casadi::DM xref(1, 1);
    // casadi::DM yref(1, 1);
    // casadi::DM li(nx, Nhor_plus_1);
    // casadi::DM ci(nx, Nhor_plus_1);
    // casadi::DM lji(nx*number_of_robots, Nhor_plus_1);
    // casadi::DM cji(nx*number_of_robots, Nhor_plus_1);
    // casadi::DM X_0(nx, 1);

    // casadi::DMVector dm_in{xref, yref, li, ci, lji, cji, X_0};
    // casadi::DMVector dm_out(ocpX_function.n_out());

    // casadi::DM x_res(1, Nhor_plus_1);
    // casadi::DM y_res(1, Nhor_plus_1);
    // casadi::DM u_res(1, Nhor);
    // casadi::DM v_res(1, Nhor);

    // xref(0,0) = 1.0;
    // yref(0,0) = 1.0;

    // // Option 2: Normal evaluation of ocpX
    // dm_out = ocpX_function(dm_in);

    // u_res = dm_out[2];
    // ROS_INFO_STREAM("Hello: " << u_res);

    // //--------------------------------------------------------------