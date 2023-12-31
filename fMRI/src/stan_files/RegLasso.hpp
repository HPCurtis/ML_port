
// Code generated by stanc v2.31.0
#include <stan/model/model_header.hpp>
namespace RegLasso_model_namespace {

using stan::model::model_base_crtp;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 42> locations_array__ = 
{" (found before start of program)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 26, column 1 to column 9)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 27, column 1 to column 24)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 28, column 1 to column 16)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 30, column 1 to column 24)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 33, column 1 to column 23)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 34, column 1 to column 25)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 35, column 1 to column 22)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 36, column 1 to column 32)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 50, column 1 to column 21)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 51, column 1 to column 51)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 40, column 1 to column 44)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 41, column 1 to column 23)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 44, column 1 to column 27)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 47, column 1 to column 37)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 4, column 1 to column 13)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 5, column 1 to column 7)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 6, column 8 to column 15)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 6, column 1 to column 25)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 7, column 8 to column 15)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 7, column 17 to column 18)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 7, column 1 to column 28)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 9, column 1 to column 12)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 10, column 8 to column 14)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 10, column 16 to column 17)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 10, column 1 to column 26)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 13, column 9 to column 16)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 13, column 18 to column 19)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 13, column 2 to column 32)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 14, column 9 to column 15)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 14, column 17 to column 18)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 14, column 2 to column 30)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 15, column 9 to column 16)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 15, column 2 to column 29)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 18, column 6 to column 80)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 19, column 6 to column 75)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 17, column 16 to line 20, column 5)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 17, column 2 to line 20, column 5)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 23, column 2 to column 55)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 28, column 8 to column 9)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 34, column 8 to column 15)",
 " (in '/home/harrison/Desktop/gitHubRepos/ML_port/fMRI/RegLasso.stan', line 50, column 13 to column 19)"};




class RegLasso_model final : public model_base_crtp<RegLasso_model> {

 private:
  int N_train;
  int p;
  Eigen::Matrix<double, -1, 1> y_train_data__;
  Eigen::Matrix<double, -1, -1> X_train_data__;
  int N_test;
  Eigen::Matrix<double, -1, -1> X_test_data__;
  Eigen::Matrix<double, -1, -1> xtrain_std_data__;
  Eigen::Matrix<double, -1, -1> xtest_std_data__;
  Eigen::Matrix<double, -1, 1> ytrain_std_data__; 
  Eigen::Map<Eigen::Matrix<double, -1, 1>> y_train{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double, -1, -1>> X_train{nullptr, 0, 0};
  Eigen::Map<Eigen::Matrix<double, -1, -1>> X_test{nullptr, 0, 0};
  Eigen::Map<Eigen::Matrix<double, -1, -1>> xtrain_std{nullptr, 0, 0};
  Eigen::Map<Eigen::Matrix<double, -1, -1>> xtest_std{nullptr, 0, 0};
  Eigen::Map<Eigen::Matrix<double, -1, 1>> ytrain_std{nullptr, 0};
 
 public:
  ~RegLasso_model() { }
  
  inline std::string model_name() const final { return "RegLasso_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.31.0", "stancflags = "};
  }
  
  
  RegLasso_model(stan::io::var_context& context__,
                 unsigned int random_seed__ = 0,
                 std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "RegLasso_model_namespace::RegLasso_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 15;
      context__.validate_dims("data initialization","N_train","int",
           std::vector<size_t>{});
      N_train = std::numeric_limits<int>::min();
      
      
      current_statement__ = 15;
      N_train = context__.vals_i("N_train")[(1 - 1)];
      current_statement__ = 16;
      context__.validate_dims("data initialization","p","int",
           std::vector<size_t>{});
      p = std::numeric_limits<int>::min();
      
      
      current_statement__ = 16;
      p = context__.vals_i("p")[(1 - 1)];
      current_statement__ = 17;
      stan::math::validate_non_negative_index("y_train", "N_train", N_train);
      current_statement__ = 18;
      context__.validate_dims("data initialization","y_train","double",
           std::vector<size_t>{static_cast<size_t>(N_train)});
      y_train_data__ = 
        Eigen::Matrix<double, -1, 1>::Constant(N_train,
          std::numeric_limits<double>::quiet_NaN());
      new (&y_train) Eigen::Map<Eigen::Matrix<double, -1, 1>>(y_train_data__.data(), N_train);
        
      
      {
        std::vector<local_scalar_t__> y_train_flat__;
        current_statement__ = 18;
        y_train_flat__ = context__.vals_r("y_train");
        current_statement__ = 18;
        pos__ = 1;
        current_statement__ = 18;
        for (int sym1__ = 1; sym1__ <= N_train; ++sym1__) {
          current_statement__ = 18;
          stan::model::assign(y_train, y_train_flat__[(pos__ - 1)],
            "assigning variable y_train", stan::model::index_uni(sym1__));
          current_statement__ = 18;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 19;
      stan::math::validate_non_negative_index("X_train", "N_train", N_train);
      current_statement__ = 20;
      stan::math::validate_non_negative_index("X_train", "p", p);
      current_statement__ = 21;
      context__.validate_dims("data initialization","X_train","double",
           std::vector<size_t>{static_cast<size_t>(N_train),
            static_cast<size_t>(p)});
      X_train_data__ = 
        Eigen::Matrix<double, -1, -1>::Constant(N_train, p,
          std::numeric_limits<double>::quiet_NaN());
      new (&X_train) Eigen::Map<Eigen::Matrix<double, -1, -1>>(X_train_data__.data(), N_train, p);
        
      
      {
        std::vector<local_scalar_t__> X_train_flat__;
        current_statement__ = 21;
        X_train_flat__ = context__.vals_r("X_train");
        current_statement__ = 21;
        pos__ = 1;
        current_statement__ = 21;
        for (int sym1__ = 1; sym1__ <= p; ++sym1__) {
          current_statement__ = 21;
          for (int sym2__ = 1; sym2__ <= N_train; ++sym2__) {
            current_statement__ = 21;
            stan::model::assign(X_train, X_train_flat__[(pos__ - 1)],
              "assigning variable X_train", stan::model::index_uni(sym2__),
                                              stan::model::index_uni(sym1__));
            current_statement__ = 21;
            pos__ = (pos__ + 1);
          }
        }
      }
      current_statement__ = 22;
      context__.validate_dims("data initialization","N_test","int",
           std::vector<size_t>{});
      N_test = std::numeric_limits<int>::min();
      
      
      current_statement__ = 22;
      N_test = context__.vals_i("N_test")[(1 - 1)];
      current_statement__ = 23;
      stan::math::validate_non_negative_index("X_test", "N_test", N_test);
      current_statement__ = 24;
      stan::math::validate_non_negative_index("X_test", "p", p);
      current_statement__ = 25;
      context__.validate_dims("data initialization","X_test","double",
           std::vector<size_t>{static_cast<size_t>(N_test),
            static_cast<size_t>(p)});
      X_test_data__ = 
        Eigen::Matrix<double, -1, -1>::Constant(N_test, p,
          std::numeric_limits<double>::quiet_NaN());
      new (&X_test) Eigen::Map<Eigen::Matrix<double, -1, -1>>(X_test_data__.data(), N_test, p);
        
      
      {
        std::vector<local_scalar_t__> X_test_flat__;
        current_statement__ = 25;
        X_test_flat__ = context__.vals_r("X_test");
        current_statement__ = 25;
        pos__ = 1;
        current_statement__ = 25;
        for (int sym1__ = 1; sym1__ <= p; ++sym1__) {
          current_statement__ = 25;
          for (int sym2__ = 1; sym2__ <= N_test; ++sym2__) {
            current_statement__ = 25;
            stan::model::assign(X_test, X_test_flat__[(pos__ - 1)],
              "assigning variable X_test", stan::model::index_uni(sym2__),
                                             stan::model::index_uni(sym1__));
            current_statement__ = 25;
            pos__ = (pos__ + 1);
          }
        }
      }
      current_statement__ = 26;
      stan::math::validate_non_negative_index("xtrain_std", "N_train",
                                              N_train);
      current_statement__ = 27;
      stan::math::validate_non_negative_index("xtrain_std", "p", p);
      current_statement__ = 28;
      xtrain_std_data__ = 
        Eigen::Matrix<double, -1, -1>::Constant(N_train, p,
          std::numeric_limits<double>::quiet_NaN());
      new (&xtrain_std) Eigen::Map<Eigen::Matrix<double, -1, -1>>(xtrain_std_data__.data(), N_train, p);
        
      
      current_statement__ = 29;
      stan::math::validate_non_negative_index("xtest_std", "N_test", N_test);
      current_statement__ = 30;
      stan::math::validate_non_negative_index("xtest_std", "p", p);
      current_statement__ = 31;
      xtest_std_data__ = 
        Eigen::Matrix<double, -1, -1>::Constant(N_test, p,
          std::numeric_limits<double>::quiet_NaN());
      new (&xtest_std) Eigen::Map<Eigen::Matrix<double, -1, -1>>(xtest_std_data__.data(), N_test, p);
        
      
      current_statement__ = 32;
      stan::math::validate_non_negative_index("ytrain_std", "N_train",
                                              N_train);
      current_statement__ = 33;
      ytrain_std_data__ = 
        Eigen::Matrix<double, -1, 1>::Constant(N_train,
          std::numeric_limits<double>::quiet_NaN());
      new (&ytrain_std) Eigen::Map<Eigen::Matrix<double, -1, 1>>(ytrain_std_data__.data(), N_train);
        
      
      current_statement__ = 37;
      for (int i = 1; i <= p; ++i) {
        current_statement__ = 34;
        stan::model::assign(xtrain_std,
          stan::math::divide(
            stan::math::subtract(
              stan::model::rvalue(X_train, "X_train",
                stan::model::index_omni(), stan::model::index_uni(i)),
              stan::math::mean(
                stan::model::rvalue(X_train, "X_train",
                  stan::model::index_omni(), stan::model::index_uni(i)))),
            stan::math::sd(
              stan::model::rvalue(X_train, "X_train",
                stan::model::index_omni(), stan::model::index_uni(i)))),
          "assigning variable xtrain_std", stan::model::index_omni(),
                                             stan::model::index_uni(i));
        current_statement__ = 35;
        stan::model::assign(xtest_std,
          stan::math::divide(
            stan::math::subtract(
              stan::model::rvalue(X_test, "X_test",
                stan::model::index_omni(), stan::model::index_uni(i)),
              stan::math::mean(
                stan::model::rvalue(X_test, "X_test",
                  stan::model::index_omni(), stan::model::index_uni(i)))),
            stan::math::sd(
              stan::model::rvalue(X_test, "X_test",
                stan::model::index_omni(), stan::model::index_uni(i)))),
          "assigning variable xtest_std", stan::model::index_omni(),
                                            stan::model::index_uni(i));
      }
      current_statement__ = 38;
      stan::model::assign(ytrain_std,
        stan::math::divide(
          stan::math::subtract(y_train, stan::math::mean(y_train)),
          stan::math::sd(y_train)), "assigning variable ytrain_std");
      current_statement__ = 39;
      stan::math::validate_non_negative_index("beta", "p", p);
      current_statement__ = 40;
      stan::math::validate_non_negative_index("linpred", "N_train", N_train);
      current_statement__ = 41;
      stan::math::validate_non_negative_index("y_test", "N_test", N_test);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = 1 + 1 + p + 1;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "RegLasso_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      local_scalar_t__ mu = DUMMY_VAR__;
      current_statement__ = 1;
      mu = in__.template read<local_scalar_t__>();
      local_scalar_t__ sigma2 = DUMMY_VAR__;
      current_statement__ = 2;
      sigma2 = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(
                 0, lp__);
      Eigen::Matrix<local_scalar_t__, -1, 1> beta =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(p, DUMMY_VAR__);
      current_statement__ = 3;
      beta = in__.template read<Eigen::Matrix<local_scalar_t__, -1, 1>>(p);
      local_scalar_t__ lambda = DUMMY_VAR__;
      current_statement__ = 4;
      lambda = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(
                 0, lp__);
      local_scalar_t__ sigma = DUMMY_VAR__;
      Eigen::Matrix<local_scalar_t__, -1, 1> linpred =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(N_train,
           DUMMY_VAR__);
      current_statement__ = 7;
      sigma = stan::math::sqrt(sigma2);
      current_statement__ = 8;
      stan::model::assign(linpred,
        stan::math::add(mu, stan::math::multiply(xtrain_std, beta)),
        "assigning variable linpred");
      current_statement__ = 5;
      stan::math::check_greater_or_equal(function__, "sigma", sigma, 0);
      {
        current_statement__ = 11;
        lp_accum__.add(
          stan::math::double_exponential_lpdf<propto__>(beta, 0,
            (sigma / lambda)));
        current_statement__ = 12;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(lambda, 0, 1));
        current_statement__ = 13;
        lp_accum__.add((-2 * stan::math::log(sigma)));
        current_statement__ = 14;
        lp_accum__.add(
          stan::math::normal_lpdf<propto__>(ytrain_std, linpred, sigma));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "RegLasso_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      double mu = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 1;
      mu = in__.template read<local_scalar_t__>();
      double sigma2 = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 2;
      sigma2 = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(
                 0, lp__);
      Eigen::Matrix<double, -1, 1> beta =
         Eigen::Matrix<double, -1, 1>::Constant(p,
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 3;
      beta = in__.template read<Eigen::Matrix<local_scalar_t__, -1, 1>>(p);
      double lambda = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 4;
      lambda = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(
                 0, lp__);
      double sigma = std::numeric_limits<double>::quiet_NaN();
      Eigen::Matrix<double, -1, 1> linpred =
         Eigen::Matrix<double, -1, 1>::Constant(N_train,
           std::numeric_limits<double>::quiet_NaN());
      out__.write(mu);
      out__.write(sigma2);
      out__.write(beta);
      out__.write(lambda);
      if (stan::math::logical_negation((stan::math::primitive_value(
            emit_transformed_parameters__) || stan::math::primitive_value(
            emit_generated_quantities__)))) {
        return ;
      } 
      current_statement__ = 7;
      sigma = stan::math::sqrt(sigma2);
      current_statement__ = 8;
      stan::model::assign(linpred,
        stan::math::add(mu, stan::math::multiply(xtrain_std, beta)),
        "assigning variable linpred");
      current_statement__ = 5;
      stan::math::check_greater_or_equal(function__, "sigma", sigma, 0);
      if (emit_transformed_parameters__) {
        out__.write(sigma);
        out__.write(linpred);
      } 
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      } 
      std::vector<double> y_test =
         std::vector<double>(N_test, 
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 10;
      stan::model::assign(y_test,
        stan::math::normal_rng(
          stan::math::add(mu, stan::math::multiply(xtest_std, beta)), sigma,
          base_rng__), "assigning variable y_test");
      out__.write(y_test);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(VecVar& params_r__, VecI& params_i__,
                                   VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      local_scalar_t__ mu = DUMMY_VAR__;
      mu = in__.read<local_scalar_t__>();
      out__.write(mu);
      local_scalar_t__ sigma2 = DUMMY_VAR__;
      sigma2 = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, sigma2);
      Eigen::Matrix<local_scalar_t__, -1, 1> beta =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(p, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= p; ++sym1__) {
        stan::model::assign(beta, in__.read<local_scalar_t__>(),
          "assigning variable beta", stan::model::index_uni(sym1__));
      }
      out__.write(beta);
      local_scalar_t__ lambda = DUMMY_VAR__;
      lambda = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, lambda);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"mu", "sigma2", "beta", "lambda",
      "sigma", "linpred", "y_test"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
      std::vector<size_t>{}, std::vector<size_t>{static_cast<size_t>(p)},
      std::vector<size_t>{}, std::vector<size_t>{},
      std::vector<size_t>{static_cast<size_t>(N_train)},
      std::vector<size_t>{static_cast<size_t>(N_test)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "mu");
    param_names__.emplace_back(std::string() + "sigma2");
    for (int sym1__ = 1; sym1__ <= p; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "beta" + '.' + std::to_string(sym1__));
      } 
    }
    param_names__.emplace_back(std::string() + "lambda");
    if (emit_transformed_parameters__) {
      param_names__.emplace_back(std::string() + "sigma");
      for (int sym1__ = 1; sym1__ <= N_train; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "linpred" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N_test; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "y_test" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "mu");
    param_names__.emplace_back(std::string() + "sigma2");
    for (int sym1__ = 1; sym1__ <= p; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "beta" + '.' + std::to_string(sym1__));
      } 
    }
    param_names__.emplace_back(std::string() + "lambda");
    if (emit_transformed_parameters__) {
      param_names__.emplace_back(std::string() + "sigma");
      for (int sym1__ = 1; sym1__ <= N_train; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "linpred" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N_test; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "y_test" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"mu\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma2\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(p) + "},\"block\":\"parameters\"},{\"name\":\"lambda\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma\",\"type\":{\"name\":\"real\"},\"block\":\"transformed_parameters\"},{\"name\":\"linpred\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_train) + "},\"block\":\"transformed_parameters\"},{\"name\":\"y_test\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N_test) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"mu\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma2\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(p) + "},\"block\":\"parameters\"},{\"name\":\"lambda\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma\",\"type\":{\"name\":\"real\"},\"block\":\"transformed_parameters\"},{\"name\":\"linpred\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_train) + "},\"block\":\"transformed_parameters\"},{\"name\":\"y_test\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N_test) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  (((1 + 1) + p) + 1);
      const size_t num_transformed = emit_transformed_parameters * 
  (1 + N_train);
      const size_t num_gen_quantities = emit_generated_quantities * N_test;
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      std::vector<int> params_i;
      vars = Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  (((1 + 1) + p) + 1);
      const size_t num_transformed = emit_transformed_parameters * 
  (1 + N_train);
      const size_t num_gen_quantities = emit_generated_quantities * N_test;
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      vars = std::vector<double>(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec(params_r.size());
      std::vector<int> params_i;
      transform_inits(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }

  inline void transform_inits(const stan::io::var_context& context,
                              std::vector<int>& params_i,
                              std::vector<double>& vars,
                              std::ostream* pstream__ = nullptr) const {
     constexpr std::array<const char*, 4> names__{"mu", "sigma2", "beta",
      "lambda"};
      const std::array<Eigen::Index, 4> constrain_param_sizes__{1, 1, p, 1};
      const auto num_constrained_params__ = std::accumulate(
        constrain_param_sizes__.begin(), constrain_param_sizes__.end(), 0);
    
     std::vector<double> params_r_flat__(num_constrained_params__);
     Eigen::Index size_iter__ = 0;
     Eigen::Index flat_iter__ = 0;
     for (auto&& param_name__ : names__) {
       const auto param_vec__ = context.vals_r(param_name__);
       for (Eigen::Index i = 0; i < constrain_param_sizes__[size_iter__]; ++i) {
         params_r_flat__[flat_iter__] = param_vec__[i];
         ++flat_iter__;
       }
       ++size_iter__;
     }
     vars.resize(num_params_r__);
     transform_inits_impl(params_r_flat__, params_i, vars, pstream__);
    } // transform_inits() 
    
};
}
using stan_model = RegLasso_model_namespace::RegLasso_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return RegLasso_model_namespace::profiles__;
}

#endif


