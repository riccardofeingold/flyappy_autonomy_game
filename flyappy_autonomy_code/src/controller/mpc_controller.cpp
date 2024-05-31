#include "flyappy_autonomy_code/controller/mpc_controller.hpp"

MPCController::MPCController(
    int Nx,
    int Nu,
    int N,
    int nWSR
) : Nx_(Nx), Nu_(Nu), N_(N), nWSR_(nWSR)
{
    Q_ = Eigen::Matrix4d::Zero();
    QBar_ = Eigen::MatrixXd::Zero(Q_.rows() * N_, Q_.cols() * N_);
    R_ = Eigen::Matrix2d::Zero();
    RBar_ = Eigen::MatrixXd::Zero(R_.rows() * N_, R_.cols() * N_);

    Sx_ = Eigen::MatrixXd::Zero(Nx_*N_, Nx_);
    Su_ = Eigen::MatrixXd::Zero(Nx_*N_, Nu_*N_);

    eigenH_ = Eigen::MatrixXd::Zero(Nu_*N, Nu_*N);
    Y_ = Eigen::MatrixXd::Zero(Nx_, Nx_);
    F_ = Eigen::MatrixXd::Zero(Nx_, Nu_*N);

    // Input Constraints
    eigenA_ = Eigen::MatrixXd::Zero(4*N, Nu_*N);
    Hu_ << 1, 0,
          -1, 0,
           0, 1,
           0,-1;
    hu_ << axUpperBound,
           -axLowerBound,
           ayUpperBound,
           -ayLowerBound;
}

MPCController::~MPCController()
{
    delete[] H_;
    delete[] g_;
    delete[] A_;
    delete[] lbA_;
    delete[] ubA_;
}

bool MPCController::solve(const Eigen::Vector4d& Xk, Eigen::Vector2d& U, bool init)
{
    // Setup consttraints
    Eigen::MatrixXd g = F_.transpose() * Xk;
    g_ = Conversions::convertEigenToRealT<double>(g);
    H_ = Conversions::convertEigenToRealT<double>(eigenH_);
    ubA_ = Conversions::convertEigenToRealT<double>(eigenUBA_);
    A_ = Conversions::convertEigenToRealT<double>(eigenA_);

    // Setup QP solver
    QProblem qp(Nu_ * N_, 4 * N_, HST_POSDEF);
    Options options;
    options.setToMPC();
    options.printLevel = PL_NONE;
    qp.setOptions(options);
    
    real_t* xOpt = new real_t[Nu_ * N_];
    
    int_t nWSR = nWSR_;
    bool success = qp.init(H_, g_, A_, nullptr, nullptr, nullptr, ubA_, nWSR) == SUCCESSFUL_RETURN && qp.getPrimalSolution(xOpt) == SUCCESSFUL_RETURN;
    if (success)
    {
        U(0) = xOpt[0];
        U(1) = xOpt[1];
    } else
    {
        U = Eigen::Vector2d(axLowerBound, 0);
    }

    delete[] xOpt;
    delete[] g_;
    delete[] H_;
    delete[] A_;
    delete[] lbA_;
    delete[] ubA_;
    g_ = nullptr;
    H_ = nullptr;
    A_ = nullptr;
    lbA_ = nullptr;
    ubA_ = nullptr;
    return success;
}

void MPCController::setStateMatrixConstraints(const Eigen::MatrixXd& Hx, const Eigen::VectorXd& hx)
{
    Hx_ = Hx;
    hx_ = hx;
}

void MPCController::setQRPMatrices(const Eigen::Matrix4d& Q, const Eigen::Matrix2d& R)
{
    Q_ = Q;
    R_ = R;
    computeP();

    // constructing all matrices need for QP
    constructQBar();
    constructRBar();
    constructSxSu();
    constructF();
    constructH();
    constructConstraintsMatrix();
    constructUpperBoundConstraints();
}

void MPCController::computeP()
{
    Eigen::Matrix4d P = Q_;
    Eigen::Matrix4d P_next;
    double tolerance = 1e-12;

    // Solve Riccati Difference Equation
    Eigen::MatrixXd Ad = system_.Ad.cast<double>();
    Eigen::MatrixXd Bd = system_.Bd.cast<double>();

    for (unsigned int i = 0; i < 1000; ++i)
    {
        Eigen::Matrix<double, 2, 4> K_temp = (Bd.transpose() * P * Bd + R_).inverse() * Bd.transpose() * P * Ad;
        P_next = Ad.transpose() * P * Ad + Q_ - Ad.transpose() * P * Bd * K_temp;

        if ((P_next - P).cwiseAbs().maxCoeff() < tolerance) 
        {
            P = P_next;
            break;
        }

        P = P_next;
    }
    P_ = P;
}

void MPCController::constructConstraintsMatrix()
{
    for (int i = 0; i < N_; ++i)
    {
        eigenA_.block(i*Nx_, i*Nu_, Nx_, Nu_) = Hu_;
    }

    A_ = Conversions::convertEigenToRealT(eigenA_);
}

void MPCController::constructUpperBoundConstraints()
{
    eigenUBA_ = Eigen::VectorXd::Zero(hu_.rows() * N_);
    
    for (int i = 0; i < N_; ++i)
    {
        eigenUBA_.block(i*hu_.rows(), 0, hu_.rows(), hu_.cols()) = hu_;
    }

    ubA_ = Conversions::convertEigenToRealT<double>(eigenUBA_);
}

void MPCController::constructQBar()
{
    for (uint8_t i = 0; i < N_ - 1; ++i)
    {
        QBar_.block(i*Q_.rows(), i*Q_.cols(), Q_.rows(), Q_.cols()) = Q_;
    }

    QBar_.block((N_ - 1)*P_.rows(), (N_ - 1)*P_.cols(), P_.rows(), P_.cols()) = P_;
}

void MPCController::constructRBar()
{
    for (uint8_t i = 0; i < N_; ++i)
    {
        RBar_.block(i*R_.rows(), i*R_.cols(), R_.rows(), R_.cols()) = R_;
    }
}

void MPCController::constructSxSu()
{
    // Sx = [A, A², A³, ..., A^N]^T
    // Su = [B 0 ... 0
    //       AB B 0 ... 0
    //       ...
    //       A^(N-1)B ... AB B]

    Eigen::Matrix4d A = system_.Ad.cast<double>();
    Eigen::Matrix<double, 4, 2> B = system_.Bd.cast<double>();

    int j = 1;
    for (int i = 0; i < N_; ++i)
    {
        Sx_.block(i*Nx_, 0, A.rows(), A.cols()) = A;
        Su_.block(i*Nx_, i*Nu_, B.rows(), B.cols()) = B;

        if (j < N_) 
        {
            int colCounter = 0;
            for (int k = j; k < N_; ++k)
            {
                Su_.block(k*Nx_, colCounter*Nu_, Nx_, Nu_) = A*B;
                colCounter++;
            }
        }

        if (i == N_) break;
        A *= system_.Ad.cast<double>();
        j++;
    }
}

void MPCController::constructH()
{
    eigenH_ = Su_.transpose() * QBar_ * Su_ + RBar_;
    H_ = Conversions::convertEigenToRealT(eigenH_);
}

void MPCController::constructF()
{
    F_ = Sx_.transpose() * QBar_ * Su_;
}

void MPCController::constructY()
{
    Y_ = Sx_.transpose() * QBar_ * Sx_;
}