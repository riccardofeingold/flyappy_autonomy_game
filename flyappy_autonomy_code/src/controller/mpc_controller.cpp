#include "flyappy_autonomy_code/controller/mpc_controller.hpp"

MPC::MPC(
    int Nx,
    int Nu,
    int N,
    int nWSR,
    bool onlyInputConstraints
) : Nx_(Nx), Nu_(Nu), N_(N), nWSR_(nWSR)
{
    onlyInputConstraints_ = onlyInputConstraints;
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
    Hu_ << 1, 0,
          -1, 0,
           0, 1,
           0,-1;
    hu_ << axUpperBound,
           -axLowerBound,
           ayUpperBound,
           -ayLowerBound;

    if (!onlyInputConstraints)
    {
        // state constraints
        Hx_ = Eigen::Matrix<double, 2, 4>::Zero();
        Hx_ << 0, -1, 0, 0,
            0,  1, 0, 0;
        hx_ = Eigen::Vector2d::Zero();
        hx_ << -VMIN, VMAX;

        // terminal constraints
        Hf_ = Eigen::MatrixXd::Zero(2*Nx_, Nx_);
        Hf_ << 
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 1, 0,
            0, 0,-1, 0,
            0, 0, 0, 1,
            0, 0, 0,-1;
        hf_ = Eigen::VectorXd::Zero(2*Nx_);
        
        eigenA_ = Eigen::MatrixXd::Zero((Hx_.rows() + Hu_.rows())*N + Hf_.rows(), Nu_*N);
    } else 
    {
        eigenA_ = Eigen::MatrixXd::Zero(Hu_.rows()*N, Nu_*N);
    }
}

MPC::~MPC()
{
    delete[] H_;
    delete[] g_;
    delete[] A_;
    delete[] lbA_;
    delete[] ubA_;
}

bool MPC::solve(const Eigen::Vector4d& Xk, const Eigen::Vector4d& Xs, const Eigen::Vector2d& Us, Eigen::Vector2d& U)
{
    // Setup consttraints
    Eigen::Vector4d deltaXk = Xk - Xs;

    // adjust upper boundary with steady input
    constructUpperBoundConstraints(Us, Xs);
    if (!onlyInputConstraints_)
    {
        eigenUBA_ += E_ * deltaXk;
    }

    Eigen::MatrixXd g = F_.transpose() * deltaXk;

    // convert to qpOASES format
    g_ = Conversions::convertEigenToRealT<double>(g);
    H_ = Conversions::convertEigenToRealT<double>(eigenH_);
    ubA_ = Conversions::convertEigenToRealT<double>(eigenUBA_);
    A_ = Conversions::convertEigenToRealT<double>(eigenA_);

    // Setup QP solver
    QProblem qp(Nu_ * N_, eigenA_.rows(), HST_POSDEF);
    Options options;
    options.setToMPC();
    options.printLevel = PL_NONE;
    qp.setOptions(options);
    
    real_t* xOpt = new real_t[Nu_ * N_];
    
    int_t nWSR = nWSR_;
    bool success = qp.init(H_, g_, A_, nullptr, nullptr, nullptr, ubA_, nWSR) == SUCCESSFUL_RETURN && qp.getPrimalSolution(xOpt) == SUCCESSFUL_RETURN;
    if (success)
    {
        U(0) = xOpt[0] + Us[0];
        U(1) = xOpt[1] + Us[1];
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

Eigen::VectorXd MPC::computeSteadyState(const Eigen::Vector4d& r)
{
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Nx_ + r.rows(), Nx_ + Nu_);
    A.block(0, 0, Nx_, Nx_) = Eigen::Matrix4d::Identity() - system_.Ad.cast<double>();
    A.block(0, Nx_, Nx_, Nu_) = -system_.Bd.cast<double>();

    A.block(Nx_, 0, 4, 4) = Eigen::Matrix4d::Identity();

    Eigen::VectorXd b = Eigen::VectorXd::Zero(Nx_ + r.rows());
    b.segment(Nx_, r.rows()) = r;

    return A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

void MPC::setStateMatrixConstraints(const Eigen::MatrixXd& Hx, const Eigen::VectorXd& hx)
{
    Hx_ = Hx;
    hx_ = hx;
}

void MPC::setTerminalMatrixConstraints(const Eigen::MatrixXd& Hf, const Eigen::VectorXd& hf)
{
    Hf_ = Hf;
    hf_ = hf;
}

void MPC::setQRPMatrices(const Eigen::Matrix4d& Q, const Eigen::Matrix2d& R)
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
    if (!onlyInputConstraints_)
    {
        constructE();
    }
    constructConstraintsMatrix();
}

void MPC::computeP()
{
    Eigen::Matrix4d P = Q_;
    Eigen::Matrix4d P_next;
    double tolerance = 1e-12;

    // Solve Riccati Difference Equation
    Eigen::MatrixXd Ad = system_.Ad.cast<double>();
    Eigen::MatrixXd Bd = system_.Bd.cast<double>();

    for (unsigned int i = 0; i < 10000; ++i)
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

void MPC::constructConstraintsMatrix()
{
    for (int i = 0; i < N_; ++i)
    {
        eigenA_.block(i*Nx_, i*Nu_, Nx_, Nu_) = Hu_;
    }

    if (!onlyInputConstraints_)
    {
        // get HxBar
        Eigen::MatrixXd HxBar = Eigen::MatrixXd::Zero((N_ - 1) * Hx_.rows(), (N_ - 1)*Hx_.cols());
        for (int i = 0; i < N_-1; ++i)
        {
            HxBar.block(i*Hx_.rows(), i*Hx_.cols(), Hx_.rows(), Hx_.cols()) = Hx_;
        }

        Eigen::MatrixXd Gx = HxBar * Su_.block(0, 0, Nx_ * (N_ - 1), Nu_ * N_);

        eigenA_.block(Hu_.rows() * N_ + Hx_.rows(), 0, Gx.rows(), Gx.cols()) = Gx;

        Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
        for (int i = N_ - 1; i > -1; --i)
        {
            eigenA_.block(eigenA_.rows() - Hf_.rows(), i*Nu_, Hf_.rows(), Nu_) = Hf_ * A * system_.Bd.cast<double>();
            A *= system_.Ad.cast<double>();
        }
    }
}

void MPC::constructUpperBoundConstraints(const Eigen::Vector2d& Us, const Eigen::Vector4d& Xs)
{
    if (onlyInputConstraints_)
    {
        eigenUBA_ = Eigen::VectorXd::Zero(hu_.rows() * N_);
        
        for (int i = 0; i < N_; ++i)
        {
            eigenUBA_.block(i*hu_.rows(), 0, hu_.rows(), hu_.cols()) = hu_ - Hu_ * Us;
        }
    }else
    {
        eigenUBA_ = Eigen::VectorXd::Zero((hu_.rows() + hx_.rows()) * N_ + hf_.rows());
        
        for (int i = 0; i < N_; ++i)
        {
            eigenUBA_.block(i*hu_.rows(), 0, hu_.rows(), hu_.cols()) = hu_ - Hu_ * Us;
        }

        for (int i = 0; i < N_; ++i)
        {
            eigenUBA_.segment(N_*hu_.rows() + i*hx_.rows(), hx_.rows()) = hx_ - Hx_ * Xs;
        }

        eigenUBA_.segment(eigenUBA_.rows() - hf_.rows(), hf_.rows()) = hf_;
    }
}

void MPC::constructQBar()
{
    for (uint8_t i = 0; i < N_ - 1; ++i)
    {
        QBar_.block(i*Q_.rows(), i*Q_.cols(), Q_.rows(), Q_.cols()) = Q_;
    }

    QBar_.block((N_ - 1)*P_.rows(), (N_ - 1)*P_.cols(), P_.rows(), P_.cols()) = P_;
}

void MPC::constructRBar()
{
    for (uint8_t i = 0; i < N_; ++i)
    {
        RBar_.block(i*R_.rows(), i*R_.cols(), R_.rows(), R_.cols()) = R_;
    }
}

void MPC::constructSxSu()
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

        if (i == N_-1) break;
        A *= system_.Ad.cast<double>();
        j++;
    }
}

void MPC::constructH()
{
    eigenH_ = Su_.transpose() * QBar_ * Su_ + RBar_;
}

void MPC::constructF()
{
    F_ = Sx_.transpose() * QBar_ * Su_;
}

void MPC::constructY()
{
    Y_ = Sx_.transpose() * QBar_ * Sx_;
}

void MPC::constructE()
{
    E_ = Eigen::MatrixXd::Zero((hu_.rows() + Hx_.rows()) * N_ + Hf_.rows(), Nx_);
    Eigen::MatrixXd HxStacked = Eigen::MatrixXd::Zero(Hx_.rows() * N_ + Hf_.rows(), Nx_);

    HxStacked.block(0, 0, Hx_.rows(), Hx_.cols()) = -Hx_;
    for (int i = 1; i < N_; ++i)
    {
        HxStacked.block(i*Hx_.rows(), 0, Hx_.rows(), Hx_.cols()) = -Hx_ * Sx_.block(i*Nx_, 0, Nx_, Nx_);
    }
    HxStacked.block(HxStacked.rows() - Hf_.rows(), 0, Hf_.rows(), Hf_.cols()) = -Hf_ * Sx_.block(Sx_.rows() - Nx_, 0, Nx_, Nx_);
    E_.block(N_*hu_.rows(), 0, HxStacked.rows(), HxStacked.cols()) = HxStacked;
}