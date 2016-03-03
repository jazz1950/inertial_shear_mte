#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_vector_base.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <Epetra_Map.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <string>

using namespace dealii;


class ParameterReader : public Subscriptor
{
public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string);

private:
    void declare_parameters();
    ParameterHandler &prm;
};

ParameterReader::ParameterReader(ParameterHandler &paramhandler)
:
prm(paramhandler)
{}


void ParameterReader::declare_parameters()
{
  prm.enter_subsection ("Mesh & geometry parameters");
    prm.declare_entry ("Gmesh input" , "false", Patterns::Bool());
    prm.declare_entry ("Input file name", "",Patterns::Anything());
    prm.declare_entry ("Given init. refinement", "0", Patterns::Integer(0,9));
    prm.declare_entry ("Number of refinement" , "3", Patterns::Integer(0,9));
    prm.declare_entry ("X-axis min" , "-0.5" , Patterns::Double(-100, 100));
    prm.declare_entry ("X-axis max" , "+0.5" , Patterns::Double(-100, 100));	
    prm.declare_entry ("Y-axis min" , "-0.5" , Patterns::Double(-100, 100));
    prm.declare_entry ("Y-axis max" , "+0.5" , Patterns::Double(-100, 100));
    prm.declare_entry ("Z-axis min" , "-0.5" , Patterns::Double(-100, 100));
    prm.declare_entry ("Z-axis max" , "+0.5" , Patterns::Double(-100, 100));
  prm.leave_subsection ();

  prm.enter_subsection ("Entropy Viscosity");
    prm.declare_entry ("Which compute viscosity", "0", Patterns::Integer(0,9));
    prm.declare_entry ("Stablization alpha", "1", Patterns::Double(0.0,100.));
    prm.declare_entry ("Beta", "1", Patterns::Double(0.0,100.));
    prm.declare_entry ("c_R factor", "1", Patterns::Double(0.0,100.));
  prm.leave_subsection ();

  prm.enter_subsection ("Boundary conditions");
    prm.declare_entry ("Inlet flow type" , "0" , Patterns::Integer(0,3));
    prm.declare_entry ("Boundary for X-axis min", "0", Patterns::Integer(0,100));
    prm.declare_entry ("Boundary for X-axis max", "0", Patterns::Integer(0,100));
    prm.declare_entry ("Boundary for Y-axis min", "0", Patterns::Integer(0,100));
    prm.declare_entry ("Boundary for Y-axis max", "0", Patterns::Integer(0,100));
    prm.declare_entry ("Boundary for Z-axis min", "0", Patterns::Integer(0,100));
    prm.declare_entry ("Boundary for Z-axis max", "0", Patterns::Integer(0,100));
  prm.leave_subsection ();

  prm.enter_subsection ("Adapative Mesh Refinement");
    prm.declare_entry ("Initial level of refinement", "0", Patterns::Integer(0,9));
    prm.declare_entry ("Level of refinement" , "0" , Patterns::Integer(0,9));
    prm.declare_entry ("Safe guard layer" , "0" , Patterns::Double(0,1000));
  prm.leave_subsection ();

  prm.enter_subsection ("Problem definition");
    
  prm.leave_subsection ();

  prm.enter_subsection ("Particle Information");
    prm.declare_entry ("DNS particle", "false" , Patterns::Bool());
    prm.declare_entry ("Number of particle" , "0" , Patterns::Integer(0,10000));
    prm.declare_entry ("Radius" , "0.0" , Patterns::Double(0.0,100));
    prm.declare_entry ("Factor for viscosity" , "0.0" , Patterns::Double(0.0,100000));
    prm.declare_entry ("Particle fraction" , "0.0" , Patterns::Double(0.0, 100));
    prm.declare_entry ("Random particles", "false" , Patterns::Bool());
    prm.declare_entry ("No flux inside" , "false" , Patterns::Bool());
    prm.declare_entry ("Threshold" , "0.0" , Patterns::Double(0.0,10.0));
  prm.leave_subsection ();

  prm.enter_subsection ("Output File Format");
    prm.declare_entry ("Solution print" , "false" , Patterns::Bool());
    prm.declare_entry ("vtu" , "false" , Patterns::Bool());
    prm.declare_entry ("gnuplot" , "false" , Patterns::Bool());
    prm.declare_entry ("tecplot" , "false" , Patterns::Bool());
  prm.leave_subsection ();

  prm.enter_subsection ("Period");
    prm.declare_entry ("output period" , "0" , Patterns::Integer(0,10000));
    prm.declare_entry ("refine period" , "-1" , Patterns::Integer(-1,10000));
  prm.leave_subsection ();

  prm.enter_subsection ("Passive Tracer");
    prm.declare_entry ("Passive tracer" , "false" , Patterns::Bool());
    prm.declare_entry ("Assigning type" , "0" , Patterns::Integer(0,10));
    prm.declare_entry ("Number of tracers" , "1000" , Patterns::Integer(0,1000000));
    prm.declare_entry ("Initial y pos" , "0.15" , Patterns::Double(-100, 100));
  prm.leave_subsection ();
  
  prm.enter_subsection ("Problem Definition");
    prm.declare_entry ("Verbal output" , "false" , Patterns::Bool());
    prm.declare_entry ("Dimension" , "2" , Patterns::Integer(1,3));
    prm.declare_entry ("Read data" , "false" , Patterns::Bool());
    prm.declare_entry ("Time interval for convection", "0.0", Patterns::Double(0, 1));
    prm.declare_entry ("Time interval for mass transfer", "0.0", Patterns::Double(0, 1));
    prm.declare_entry ("Final time" , "1.0" , Patterns::Double(0.0,100000000));
    prm.declare_entry ("Density of particle" , "0.5" , Patterns::Double(0.0, 100000));
    prm.declare_entry ("Reynold number" , "1.0" , Patterns::Double(0, 100000));
    prm.declare_entry ("Peclet number" , "1.0" , Patterns::Double(0, 100000000));
    prm.declare_entry ("Mean velocity" , "1.0" , Patterns::Double(0, 100000));
    prm.declare_entry ("Outer error for Navier-Stokes" , "1e-06" , Patterns::Double(0,1));
    prm.declare_entry ("Outer error for mass transfer" , "1e-11" , Patterns::Double(0,1));
    prm.declare_entry ("Convective flow" , "false" , Patterns::Bool());
    prm.declare_entry ("Mass transfer" , "false" , Patterns::Bool());
  prm.leave_subsection ();
}

void ParameterReader::read_parameters(const std::string parameter_file)
{
  declare_parameters();
  prm.read_input (parameter_file);
}

template <int dim>
class Concentration : public Function<dim>
{

  public:

    Concentration ();
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &p,
                                    std::vector<Vector<double> > &values) const;

};

template <int dim>
Concentration<dim>::Concentration () :
Function<dim>(1)
{}

template <int dim>
double Concentration<dim>::value (const Point<dim>  &p,
				  const unsigned int component) const
{
  double zz = 0.0;
  if (std::abs(p[1]-0.5)<1e-5) zz = 10000.0;
//   if (p[1]>0.001) zz = 10000.0;
  return zz;
}

template <int dim>
void Concentration<dim>::vector_value (const Point<dim> &p,
					Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values(c) = Concentration<dim>::value (p, c);
}

template <int dim>
void Concentration<dim>::vector_value_list (const std::vector<Point<dim> > &points,
					    std::vector<Vector<double> >   &value_list) const
{
  for (unsigned int p=0; p<points.size(); ++p)
    Concentration<dim>::vector_value (points[p], value_list[p]);
}

template <int dim>
class Inflow_Velocity : public Function<dim>
{

  public:

    Inflow_Velocity (double, unsigned int);
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &p,
                                    std::vector<Vector<double> > &values) const;

    double init_mean_vel;
    
    unsigned int which_inflow_type;
};

template <int dim>
Inflow_Velocity<dim>::Inflow_Velocity (double init_mean_vel,
					unsigned int which_inflow_type) :
Function<dim> (dim),
init_mean_vel (init_mean_vel),
which_inflow_type (which_inflow_type)
{}

template <int dim>
double Inflow_Velocity<dim>::value (const Point<dim>  &p,
                                    const unsigned int component) const
{
    double zz = 0.0;
    const double H  = 1.0;
    
    if (component == 0 && which_inflow_type == 0) zz = init_mean_vel;
    if (component == 0 && which_inflow_type == 1) zz = 4.*init_mean_vel*p(1)*(H - p(1))/(H*H);
    if (component == 0 && which_inflow_type == 2) zz = init_mean_vel*(p[1] + 0.0);
   
    return zz;
}

template <int dim>
void Inflow_Velocity<dim>::vector_value (const Point<dim> &p,
                                         Vector<double>   &values) const
{
	for (unsigned int c=0; c<this->n_components; ++c)
		values(c) = Inflow_Velocity<dim>::value (p, c);
}

template <int dim>
void Inflow_Velocity<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                 std::vector<Vector<double> >   &value_list) const
{
	for (unsigned int p=0; p<points.size(); ++p)
		Inflow_Velocity<dim>::vector_value (points[p], value_list[p]);
}

template <int dim>
class Re_MTE_At_UBC
{
public:
    Re_MTE_At_UBC (ParameterHandler &);
    ~Re_MTE_At_UBC ();
    void run ();

private:

    void readat ();
    void create_triangulation ();

    void setup_dofs (bool, bool, bool);
    void setup_dofs_con (bool, bool);
    
    void impose_bnd_indicator_flow ();
    void impose_bnd_indicator_mass ();
    
    void boundary_ind_noFlux_c ();

    void get_initial_flow_state ();
    void find_the_streamlines ();
    
    void particle_distribution ();
    void extrapolation_step ();
    void diffusion_step ();
    void projection_step ();
    void pressure_correction_step_rot ();
    void pressure_correction_step_stn ();
    void solution_update ();
    void vel_pre_convergence ();
    
    void transfer_velocities_on_concentr ();
    void assemble_matrix_for_mass ();
    void assemble_rhs_vector_for_mass ();
    void concentr_solve ();
    
    void compute_vorticity ();
    std::pair<unsigned int, double> distant_from_particles (Point<dim> &);

    void plotting_solution_flow (unsigned int);
    void plotting_solution_mass (unsigned int);
    
    void particle_generation ();
    void pars_move (std::ofstream &, std::ofstream &);
    void pars_angular (std::ofstream &);
  
    void assigning_tracer_distribution ();
    void plotting_tracers (unsigned int);
    void find_fluid_veloicity_at_tracer_position ();
    void advancement_for_tracer ();
    void bi_section_search ( Point<dim> &, 
			      Point<dim> &,
			      std::vector<Point<dim> > &,
			      Point<dim> &);
    
    void particle_refine_mesh ();
    void refine_mesh (bool );
    void prescribe_flag_refinement ();
    void adp_execute_transfer (bool);

    void compute_quantities (std::ofstream &, double);
    
    double get_maximal_velocity () const;
    std::pair<double,double> get_extrapolated_concentr_range ();
    double get_entropy_variation_concentr (const double average_concentr) const;

    double
    compute_viscosity(  const std::vector<double>  		&,
			  const std::vector<double>          	&,
			  const std::vector<Tensor<1,dim> >  	&,
			  const std::vector<Tensor<1,dim> >  	&,
			  const std::vector<double>          	&,
			  const std::vector<double>         	&,
			  const std::vector<Tensor<1,dim> >  	&,
			  const std::vector<Tensor<1,dim> >  	&,
			  const std::vector<double>          	&,
			  const double,
			  const double,
			  const double,
			  const double,
			  const double,
			  double, double);
    
    void make_periodicity_constraints (DoFHandler<dim>	&,
					types::boundary_id  	,
					types::boundary_id  	,
					int                 	,
					ConstraintMatrix	&);
    
    void make_periodicity_constraints (
				  const typename DoFHandler<dim>::face_iterator    &face_1,
				  const typename identity<typename DoFHandler<dim>::face_iterator>::type &face_2,
				  ConstraintMatrix		&constraint_matrix,
				  const ComponentMask		&component_mask,
				  const bool			face_orientation,
				  const bool			face_flip,
				  const bool			face_rotation);
    
    void set_periodicity_constraints (const typename DoFHandler<dim>::face_iterator                    	&face_1,
                                 const typename identity<typename DoFHandler<dim>::face_iterator>::type	&face_2,
                                 const FullMatrix<double>                    				&transformation,
                                 ConstraintMatrix                    					&constraint_matrix,
                                 const ComponentMask                         				&component_mask,
                                 const bool                                   			face_orientation,
                                 const bool                                   			face_flip,
                                 const bool                                   			face_rotation);


    const Epetra_Comm			&trilinos_communicator;
    ConditionalOStream			pcout;
    ParameterHandler			&prm;

    Triangulation<dim>			triangulation, triangulation_concentr;
    double				global_Omega_diameter;

    const FESystem<dim>			fe_velocity, fe_velocity_on_concentr;
    FE_Q<dim>				fe_pressure, fe_concentr;

    DoFHandler<dim>			dof_handler_velocity;
    DoFHandler<dim>			dof_handler_velocity_on_concentr;
    DoFHandler<dim>			dof_handler_pressure;
    DoFHandler<dim>			dof_handler_concentr;

    ConstraintMatrix			constraint_velocity;
    ConstraintMatrix			constraint_pressure;
    ConstraintMatrix			constraints_concentr;

    TrilinosWrappers::SparseMatrix	matrix_velocity;
    TrilinosWrappers::SparseMatrix	matrix_pressure;
    TrilinosWrappers::SparseMatrix	matrix_concentr;

    TrilinosWrappers::Vector		vel_star;
    TrilinosWrappers::Vector		vel_n_plus_1, vel_n, vel_n_minus_1;
    
    TrilinosWrappers::Vector		vel_nPlus_con, vel_n_con;
    
    TrilinosWrappers::Vector		pre_n_plus_1, pre_n;
    TrilinosWrappers::Vector		aux_n_plus_1, aux_n, aux_n_minus_1;
    
    TrilinosWrappers::Vector		vorticity, particle_distb;
    TrilinosWrappers::Vector		vel_res, pre_res;
    
    TrilinosWrappers::Vector		concentr_solution, truc_concentration;
    TrilinosWrappers::Vector		old_concentr_solution, tmp_concentr_solution;
    TrilinosWrappers::Vector		stream_function;

    TrilinosWrappers::MPI::Vector	rhs_velocity;
    TrilinosWrappers::MPI::Vector	rhs_pressure;
    TrilinosWrappers::MPI::Vector	rhs_concentr;

    // Basic
    bool reaDat;
    double error_crit_NS, error_crit_MassTrn;
    double time_step, old_time_step, final_time, time_step_mtn, given_time_step;
    unsigned int timestep_number, Inlet_velocity_type;
    double Reynolds_number, Peclet_number, mean_velocity, maximal_velocity;
    bool is_verbal_output, turn_on_periodic_bnd, is_mass_trn, is_convective_flow;
    bool is_passive_tracer;
    double pressure_l2_norm;
    bool is_solutionPrint, ivtu, iGnuplot, iTecplot;
    unsigned int output_fac, refine_fac;
    TimerOutput computing_timer;
    
    //Particle
    bool is_dns_par;
    bool is_solid_particle;
    unsigned int num_pars;
    double par_fraction;
    bool is_random_particles;
    bool is_no_flux_concentr;
    double density_of_particle;
    double par_rad;
    double FacPar;
    double a_raTStre;
    double asp_rat_cylin;
    double ori_x, ori_y, ori_z;
    std::vector<Point<dim> > cenPar, image_cenPar;
    std::vector<bool> is_solid_inside;
    double thr_val_particle_dist;
    
    //Coarse Mesh
    bool iGmsh;
    std::string input_mesh_file;
    unsigned int grNum, given_grNum_Gmsh;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double h_size;
    int xmin_bnd, xmax_bnd, ymin_bnd, ymax_bnd, zmin_bnd, zmax_bnd; 
    unsigned int ini_reLev, reLev, max_level, min_level;
    std::vector<int> ParsRef;
    double safe_fac;
    
    //For_Entropy_Viscosity
    double stabilization_beta, stabilization_c_R;
    unsigned int stabilization_alpha;
    
    //Passive Tracer
    unsigned int no_assign_type;
    unsigned int num_of_tracers;
    double threshold_ypos, touch_at_ymax, touch_at_ymin;
    std::vector<unsigned int> what_type_tracer;
    std::vector<Point<dim> > tracer_position, old_tracer_position, image_tracer_position;
    std::vector<Point<dim> > tracer_velocity, old_tracer_velocity;
    std::vector<Point<dim> > fluid_velocity_at_tracer;
    
    
};

template <int dim>
Re_MTE_At_UBC<dim>::Re_MTE_At_UBC (ParameterHandler &param)
:
trilinos_communicator (Utilities::Trilinos::comm_world()),
pcout (std::cout, Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)==0),
prm(param),
fe_velocity (FE_Q<dim>(2), dim),
fe_velocity_on_concentr (FE_Q<dim>(2), dim),
fe_pressure (2),
fe_concentr (2),
dof_handler_velocity (triangulation),
dof_handler_velocity_on_concentr (triangulation_concentr),
dof_handler_pressure (triangulation),
dof_handler_concentr (triangulation_concentr),
time_step (0),
old_time_step (0),
timestep_number (0),
turn_on_periodic_bnd (false),
computing_timer (pcout, TimerOutput::summary,TimerOutput::wall_times),
ParsRef(100000),
touch_at_ymax (0.0),
touch_at_ymin (0.0)
{}

template <int dim>
Re_MTE_At_UBC<dim>::~Re_MTE_At_UBC ()
{
    dof_handler_velocity.clear ();
    dof_handler_velocity_on_concentr.clear (); 
    dof_handler_pressure.clear ();
    dof_handler_concentr.clear ();
}

template <int dim>
void Re_MTE_At_UBC<dim>::readat ()
{  
  pcout << "* Read Data.. " << std::endl;

  prm.enter_subsection ("Problem Definition");
    is_verbal_output = prm.get_bool ("Verbal output");
    reaDat = prm.get_bool ("Read data");
    time_step = prm.get_double ("Time interval for convection");
    time_step_mtn = prm.get_double ("Time interval for mass transfer");
    final_time = prm.get_double ("Final time");
    density_of_particle = prm.get_double ("Density of particle");
    Reynolds_number = prm.get_double ("Reynold number");
    Peclet_number = prm.get_double ("Peclet number");
    mean_velocity = prm.get_double ("Mean velocity");
    error_crit_NS = prm.get_double ("Outer error for Navier-Stokes");
    error_crit_MassTrn = prm.get_double ("Outer error for mass transfer");    
    is_convective_flow = prm.get_bool ("Convective flow");
    is_mass_trn = prm.get_bool ("Mass transfer");
  prm.leave_subsection ();

  prm.enter_subsection ("Passive Tracer");
    is_passive_tracer = prm.get_bool ("Passive tracer");
    no_assign_type = prm.get_integer ("Assigning type");
    num_of_tracers = prm.get_integer ("Number of tracers");
    threshold_ypos = prm.get_double ("Initial y pos");
  prm.leave_subsection ();
  
  prm.enter_subsection ("Period");
    output_fac = prm.get_integer ("output period");
    refine_fac = prm.get_integer ("refine period");
  prm.leave_subsection ();

  prm.enter_subsection ("Output File Format");
    is_solutionPrint = prm.get_bool ("Solution print");
    ivtu = prm.get_bool ("vtu");
    iGnuplot = prm.get_bool ("gnuplot");
    iTecplot = prm.get_bool ("tecplot");
  prm.leave_subsection ();
    
  prm.enter_subsection ("Particle Information");
    is_dns_par = prm.get_bool ("DNS particle");
    num_pars = prm.get_integer ("Number of particle");
    par_rad = prm.get_double ("Radius");
    FacPar = prm.get_double ("Factor for viscosity");
    is_random_particles = prm.get_bool ("Random particles");
    par_fraction = prm.get_double ("Particle fraction");
    is_no_flux_concentr = prm.get_bool ("No flux inside");
    thr_val_particle_dist = prm.get_double ("Threshold");
  prm.leave_subsection ();

  prm.enter_subsection ("Mesh & geometry parameters");
    iGmsh = prm.get_bool ("Gmesh input");
    input_mesh_file = prm.get ("Input file name");
    given_grNum_Gmsh = prm.get_integer ("Given init. refinement");
    xmin = prm.get_double ("X-axis min");
    xmax = prm.get_double ("X-axis max");
    ymin = prm.get_double ("Y-axis min");
    ymax = prm.get_double ("Y-axis max");
    zmin = prm.get_double ("Z-axis min");
    zmax = prm.get_double ("Z-axis max");
  prm.leave_subsection ();

  prm.enter_subsection ("Boundary conditions");
    Inlet_velocity_type = prm.get_integer ("Inlet flow type");
    xmin_bnd = prm.get_integer ("Boundary for X-axis min");
    xmax_bnd = prm.get_integer ("Boundary for X-axis max");
    ymin_bnd = prm.get_integer ("Boundary for Y-axis min");
    ymax_bnd = prm.get_integer ("Boundary for Y-axis max");
    zmin_bnd = prm.get_integer ("Boundary for Z-axis min");
    zmax_bnd = prm.get_integer ("Boundary for Z-axis max");
  prm.leave_subsection ();

  prm.enter_subsection ("Adapative Mesh Refinement");
    ini_reLev = prm.get_integer ("Initial level of refinement");
    reLev = prm.get_integer ("Level of refinement");
    safe_fac = prm.get_double ("Safe guard layer");
  prm.leave_subsection ();
  
  if (iGmsh == true)
  {
    max_level = reLev+ini_reLev;
    min_level = 0;
    h_size = std::abs(xmax-xmin)/std::pow(2.0, double(given_grNum_Gmsh+max_level));
  } 
  else 
  {
    max_level = ini_reLev+reLev;
    min_level = ini_reLev;
    h_size = std::abs(xmax-xmin)/std::pow(2.0, double(max_level));
  }
  
  given_time_step = time_step;
  old_time_step = time_step;
   
  time_step = time_step/mean_velocity;
  Reynolds_number = Reynolds_number/mean_velocity;
  
  pcout << "## Modified Time Step = " << time_step << std::endl;
  pcout << "## Modified Reynolds_number = " << Reynolds_number << std::endl;
  
  is_solid_particle = false;
  if (num_pars > 0 && par_rad >0) is_solid_particle = true;
  
  turn_on_periodic_bnd = false;
 
  for (unsigned int i=0; i<ParsRef.size(); ++i)
    ParsRef[i] = 0;
}

template <int dim>
void Re_MTE_At_UBC<dim>::create_triangulation ()
{
    if (is_verbal_output == true) 
      pcout << "* Create Triangulation.." << std::endl;

    if (iGmsh == false)
    {
      Point<dim> p1, p2; p1[0]=xmin; p1[1]=ymin; p2[0]=xmax; p2[1]=ymax; 
      GridGenerator::hyper_rectangle (triangulation,p1,p2,false);
    }
    else if(iGmsh) 
    {
      GridIn<dim> gmsh_input;
      std::ifstream in(input_mesh_file.c_str());
      gmsh_input.attach_triangulation (triangulation);
      gmsh_input.read_msh (in);
    }
    
    triangulation_concentr.copy_triangulation (triangulation);
    triangulation.refine_global (ini_reLev);
    triangulation_concentr.refine_global (ini_reLev+reLev);
}


template <int dim>
void Re_MTE_At_UBC<dim>::impose_bnd_indicator_flow ()
{
    for (typename Triangulation<dim>::active_cell_iterator
         cell=triangulation.begin_active();
         cell!=triangulation.end(); ++cell)
    {
        const Point<dim> cell_center = cell->center();

        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
            const Point<dim> face_center = cell->face(f)->center();

            if (cell->face(f)->at_boundary())
            {
		if (Inlet_velocity_type == 1)
		  cell->face(f)->set_boundary_indicator (0);

		if (std::abs(face_center[0]-(xmin)) < 1e-6)
		    cell->face(f)->set_boundary_indicator (xmin_bnd);
		
		if (std::abs(face_center[0]-(xmax)) < 1e-6)
		    cell->face(f)->set_boundary_indicator (xmax_bnd);
		
		if (std::abs(face_center[1]-(ymin)) < 1e-6)
		    cell->face(f)->set_boundary_indicator (ymin_bnd);
		
		if (std::abs(face_center[1]-(ymax)) < 1e-6)
		    cell->face(f)->set_boundary_indicator (ymax_bnd);
	    }
	}
    }  
}

template <int dim>
void Re_MTE_At_UBC<dim>::impose_bnd_indicator_mass ()
{
    for (typename Triangulation<dim>::active_cell_iterator
         cell=triangulation_concentr.begin_active();
         cell!=triangulation_concentr.end(); ++cell)
    {
        const Point<dim> cell_center = cell->center();

        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
            const Point<dim> face_center = cell->face(f)->center();

            if (cell->face(f)->at_boundary())
            {
		
		if (std::abs(face_center[0]-(xmin)) < 1e-6)
		    cell->face(f)->set_boundary_indicator (xmin_bnd);
		
		if (std::abs(face_center[0]-(xmax)) < 1e-6)
		    cell->face(f)->set_boundary_indicator (xmax_bnd);
		
		if (std::abs(face_center[1]-(ymin)) < 1e-6)
		    cell->face(f)->set_boundary_indicator (ymin_bnd);
		
		if (std::abs(face_center[1]-(ymax)) < 1e-6)
		    cell->face(f)->set_boundary_indicator (ymax_bnd);
	    }
	}
    }  
}

template <int dim>
void Re_MTE_At_UBC<dim>::setup_dofs (bool on_coarse_mesh,
				     bool is_init_vector,
				     bool is_init_matrix)
{
    pcout << "* Setup Dofs.. UPL" << ", " << is_init_matrix << ", "
	  << Utilities::Trilinos::get_n_mpi_processes(trilinos_communicator)
	  << ", ";

    GridTools::partition_triangulation (Utilities::Trilinos::get_n_mpi_processes(trilinos_communicator),
                                      triangulation);

    pcout  << triangulation.n_active_cells()
	   << ", "
	   << triangulation.n_levels()
	   << std::endl;
	
    impose_bnd_indicator_flow ();
    
    //velocity
    {
	dof_handler_velocity.distribute_dofs (fe_velocity);
	DoFRenumbering::subdomain_wise (dof_handler_velocity);

	constraint_velocity.clear ();

	DoFTools::make_hanging_node_constraints (dof_handler_velocity,
 						   constraint_velocity);
		
	if (turn_on_periodic_bnd)
	make_periodicity_constraints (dof_handler_velocity,
					3,
					4,
					0,
					constraint_velocity);
	
	constraint_velocity.close ();
	
	unsigned int n_u = dof_handler_velocity.n_dofs();
	unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association 
				   (dof_handler_velocity,
                                  Utilities::Trilinos::get_this_mpi_process
                                  (trilinos_communicator));

	Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

// 	std::cout << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)
// 		  << " = "
// 		  << n_u << " " << local_dofs << std::endl;

	if (is_init_vector)
	{
	  vel_star.reinit (map);
	  vel_n.reinit (map);
	  vel_n_plus_1.reinit (map);
	  vel_n_minus_1.reinit (map);
	  vel_res.reinit (map);
	  rhs_velocity.reinit(map);
	}
	
	if (is_init_matrix == true)
	{
	    matrix_velocity.clear();
	    TrilinosWrappers::SparsityPattern sp (map);

	    DoFTools::make_sparsity_pattern (dof_handler_velocity, sp,
					      constraint_velocity, false,
					      Utilities::Trilinos::
					      get_this_mpi_process(trilinos_communicator));
	    sp.compress();
// 	    std::cout << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)
//                  <<  " th For U" << " "
//                  << sp.n_nonzero_elements()  << std::endl;
	    matrix_velocity.reinit (sp);
	}
    }

    //pressure
    {
	dof_handler_pressure.distribute_dofs (fe_pressure);
	DoFRenumbering::subdomain_wise (dof_handler_pressure);

	constraint_pressure.clear ();
	DoFTools::make_hanging_node_constraints (dof_handler_pressure,
						  constraint_pressure);
	
	if (turn_on_periodic_bnd)
	make_periodicity_constraints (dof_handler_pressure,
					3,
					4,
					0,
					constraint_pressure);
	

// 	if (turn_on_periodic_bnd)
// 	DoFTools::make_periodicity_constraints (dof_handler_pressure,
// 					6,
// 					7,
// 					1,
// 					constraint_pressure);
	
	constraint_pressure.close ();

	unsigned int n_p = dof_handler_pressure.n_dofs();
	unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association 
				   (dof_handler_pressure,
                                  Utilities::Trilinos::get_this_mpi_process
                                  (trilinos_communicator));

	Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
// 	std::cout << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)
// 		  << " = "
// 		  << n_p << " " << local_dofs << std::endl;
	if (is_init_vector)
	{	
	  pre_n_plus_1.reinit (map);
	  pre_n.reinit (map);
	  aux_n_plus_1.reinit (map);
	  aux_n.reinit (map);
	  aux_n_minus_1.reinit (map);
	
	  pre_res.reinit (map);
	  rhs_pressure.reinit(map);
	  vorticity.reinit (map);
	  stream_function.reinit (map);
	  particle_distb.reinit (map);
	}
	
	if (is_init_matrix == true)
	{
	    matrix_pressure.clear();
	    TrilinosWrappers::SparsityPattern sp (map);

	    DoFTools::make_sparsity_pattern (dof_handler_pressure, sp,
					      constraint_pressure, false,
					      Utilities::Trilinos::
					      get_this_mpi_process(trilinos_communicator));
	    sp.compress();
	    
// 	    std::cout << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)
//                  <<  " th For P" << " "
//                  << sp.n_nonzero_elements()  << std::endl;
		 
	    //std::ostringstream filename;
	    //filename << "sp" << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator);
	    //std::ofstream output (filename.str().c_str());
	    //sp.print_gnuplot (output);
	    matrix_pressure.reinit (sp);
	}
    }
}


template <int dim>
void Re_MTE_At_UBC<dim>::setup_dofs_con (bool on_coarse_mesh, bool is_init_matrix)
{
    pcout << "* Setup Dofs.. C" << ", ";

    GridTools::partition_triangulation (Utilities::Trilinos::get_n_mpi_processes
					(trilinos_communicator),
					triangulation_concentr);

    pcout  << triangulation_concentr.n_active_cells()
	   << ", "
	   << triangulation_concentr.n_levels()
	   << std::endl;

    impose_bnd_indicator_mass ();
    
    // concentration
    {
	dof_handler_concentr.distribute_dofs (fe_concentr);
	DoFRenumbering::subdomain_wise (dof_handler_concentr);

	constraints_concentr.clear ();
	DoFTools::make_hanging_node_constraints (dof_handler_concentr,
						   constraints_concentr);
	if (turn_on_periodic_bnd)
 	make_periodicity_constraints (dof_handler_concentr,
 					3,
 					4,
 					0,
 					constraints_concentr);
	
	constraints_concentr.close ();

	unsigned int n_c = dof_handler_concentr.n_dofs();
	unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_concentr,
				Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));

	Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

	concentr_solution.reinit (map);
	rhs_concentr.reinit (map);
	old_concentr_solution.reinit (map);
	tmp_concentr_solution.reinit (map);
	truc_concentration.reinit (map);

	if (is_init_matrix == true)
	{
	    TrilinosWrappers::SparsityPattern sp (map);
	    DoFTools::make_sparsity_pattern (dof_handler_concentr, sp,
				      constraints_concentr, false,
				      Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
	    sp.compress();
	    
// 	    std::cout << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)
//                  <<  " th For C" << " "
//                  << sp.n_nonzero_elements()  << std::endl;
		 
	    matrix_concentr.reinit (sp);
	}
    }

    //velocity_on_concentr
    {
	dof_handler_velocity_on_concentr.distribute_dofs (fe_velocity_on_concentr);
	DoFRenumbering::subdomain_wise (dof_handler_velocity_on_concentr);

	unsigned int n_v_con = dof_handler_velocity_on_concentr.n_dofs();
	unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_velocity_on_concentr,
				Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
	Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

	vel_nPlus_con.reinit (map);
	vel_n_con.reinit (map);
    }

    pcout << std::endl;
}

template <int dim>
void Re_MTE_At_UBC<dim>::boundary_ind_noFlux_c ()
{
    double h_size = 0.0;
    for (typename Triangulation<dim>::active_cell_iterator
	cell=triangulation_concentr.begin_active();
	cell!=triangulation_concentr.end(); ++cell)
    {
	Point<dim> cell_center = cell->center();

	std::pair<unsigned int,double> distant_of_par = distant_from_particles (cell_center);

        if (distant_of_par.second <0.0 - 0.707*h_size)
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
            const Point<dim> face_center = cell->face(f)->center();

	    cell->face(f)->set_boundary_indicator (5);
        }
    }
}

template <int dim>
std::pair<unsigned int, double>
Re_MTE_At_UBC<dim>::distant_from_particles (Point<dim> &coor)
{
    unsigned int q1 = std::numeric_limits<unsigned int>::max();
    double q2 = std::numeric_limits<double>::max();

	for (unsigned int n = 0 ; n < num_pars ; ++n)
	{
	    double tt = cenPar[n].distance(coor) - par_rad;
	    double qq = image_cenPar[n].distance(coor) - par_rad;
	    if (std::min (tt, qq) < q2)
	      {q1 = n; q2 = std::min(std::min (tt, qq), q2);}
	}

    return std::make_pair(q1, q2);  
}

template <int dim>
void Re_MTE_At_UBC<dim>::particle_distribution ()
{
    pcout << "* Particle Distribution.. ";

    matrix_pressure = 0;
    rhs_pressure = 0;

    const QGauss<dim> quadrature_formula(fe_pressure.get_degree()+1 + 1);

    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);

    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_pressure.begin_active(),
    endc = dof_handler_pressure.end();

    for (; cell!=endc; ++cell)
        if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
        {
            fe_values_pressure.reinit (cell);
            cell->get_dof_indices (local_dofs_indices);

            local_matrix = 0;
            local_rhs = 0;

	    Point<dim> c = cell->center();
	    std::pair<unsigned int,double> distant_of_par = distant_from_particles (c);
	
	    double q2 = distant_of_par.second;
	    double tt_d = 1./(1. + std::exp( (q2)/(0.5*h_size)));
	    
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
		    local_matrix(i,j)	+=  fe_values_pressure.shape_value(i, q)*
					    fe_values_pressure.shape_value(j, q)*
					    fe_values_pressure.JxW(q);

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    local_rhs(i) +=	fe_values_pressure.shape_value(i, q)*
					tt_d*
					fe_values_pressure.JxW(q);

            }

            constraint_pressure.distribute_local_to_global (local_matrix,
                                                            local_dofs_indices,
                                                            matrix_pressure);

            constraint_pressure.distribute_local_to_global (local_rhs,
                                                            local_dofs_indices,
                                                            rhs_pressure);
        }


    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_pressure,
			      Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::MPI::Vector distibuted_xx (map);

    matrix_pressure.compress(VectorOperation::add);
    rhs_pressure.compress(VectorOperation::add);

    SolverControl solver_control (matrix_pressure.m(), error_crit_NS*rhs_pressure.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionILU preconditioner;
    preconditioner.initialize (matrix_pressure);
    cg.solve (matrix_pressure, distibuted_xx, rhs_pressure, preconditioner);
    particle_distb = distibuted_xx;

    pcout   << solver_control.last_step()
            << std::endl;

    constraint_pressure.distribute (particle_distb);  
    
//   std::vector<Point<dim> > sp (dof_handler_pressure.n_dofs());
//   MappingQ<dim> ff(fe_pressure.get_degree());
//   DoFTools::map_dofs_to_support_points (ff, dof_handler_pressure, sp);
        
//   for (unsigned int i=0; i<particle_distb.size(); ++i)
//   {  
//     unsigned int q1 = std::numeric_limits<unsigned int>::max();
//     double q2 = std::numeric_limits<double>::max();
  
//     for (unsigned int n=0; n<num_pars; ++n)
//     {
//       double tt = cenPar[n].distance(sp[i]);
//       double qq = image_cenPar[n].distance(sp[i]);
//       if (std::min (tt, qq) < q2)
//       {
// 	q1 = n; 
// 	q2 = std::min(std::min (tt, qq), q2);
// 	double tt_d = 1./(1. + std::exp( (q2-par_rad)/0.0008*h_size));
// 	particle_distb (i) = tt_d;
//       }
//     }
//   }
}

template <int dim>
void Re_MTE_At_UBC<dim>::make_periodicity_constraints (DoFHandler<dim>	&dof_handler,
							types::boundary_id  	b_id1,
							types::boundary_id  	b_id2,
							int                 	direction,
							ConstraintMatrix	&constraint_matrix)
{
    pcout << "* Make Periodicity... " << static_cast<int>(b_id1) 
	  << " | " << static_cast<int>(b_id2)<< std::endl;
    typedef typename DoFHandler<dim>::face_iterator FaceIterator;
    typedef std::map<FaceIterator, std::pair<FaceIterator, std::bitset<3> > > FaceMap;
    Tensor<1, dim> offset;

    std::set<typename DoFHandler<dim>::face_iterator> faces1;
    std::set<typename DoFHandler<dim>::face_iterator> faces2;

    for (typename DoFHandler<dim>::cell_iterator 
	  cell = dof_handler.begin();
	  cell != dof_handler.end(); ++cell)
      {
        for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
          {
            const typename DoFHandler<dim>::face_iterator face = cell->face(i);

            if (face->at_boundary() && face->boundary_indicator() == b_id1)
	    {
// 	      pcout << "1 = " << face->center() << std::endl;
              faces1.insert(face);
	    }
	    

            if (face->at_boundary() && face->boundary_indicator() == b_id2)
	    {
// 	      pcout << "2 = " << face->center() << std::endl;
              faces2.insert(face);
	    }
          }
      }

    Assert (faces1.size() == faces2.size(),
            ExcMessage ("Unmatched faces on periodic boundaries"));
  
  typedef std::pair<FaceIterator, std::bitset<3> > ResultPair;
  std::map<FaceIterator, ResultPair> matched_faces;
    
    // Match with a complexity of O(n^2). This could be improved...
    std::bitset<3> orientation;
    
    pcout << "* Match Faces... " << std::endl;
    typedef typename std::set<FaceIterator>::const_iterator SetIterator;
    for (SetIterator it1 = faces1.begin(); it1 != faces1.end(); ++it1)
      {
        for (SetIterator it2 = faces2.begin(); it2 != faces2.end(); ++it2)
          {
            if (GridTools::orthogonal_equality(orientation, *it1, *it2,
                                               direction, offset))
              {
// 		pcout << *it1 << " | " << *it2 << std::endl;
                // We have a match, so insert the matching pairs and
                // remove the matched cell in faces2 to speed up the
                // matching:
                matched_faces[*it1] = std::make_pair(*it2, orientation);
                faces2.erase(it2);
                break;
              }
          }
      }

      pcout << "* Insert Constraints... " << std::endl;
      FEValuesExtractors::Vector velocities(0);
      const ComponentMask component_mask = fe_velocity.component_mask (velocities);
      
      for (typename FaceMap::iterator it = matched_faces.begin();
         it != matched_faces.end(); ++it)
      {
        typedef typename DoFHandler<dim>::face_iterator FaceIterator;
        const FaceIterator &face_1 = it->first;
        const FaceIterator &face_2 = it->second.first;
        const std::bitset<3> &orientation = it->second.second;

        Assert(face_1->at_boundary() && face_2->at_boundary(),
               ExcInternalError());

        Assert (face_1->boundary_indicator() == b_id1 &&
                face_2->boundary_indicator() == b_id2,
                ExcInternalError());

        Assert (face_1 != face_2,
                ExcInternalError());

// 	pcout << face_1 << std::endl;
	make_periodicity_constraints (face_1,
				      face_2,
				      constraint_matrix,
				      component_mask,
				      orientation[0],
				      orientation[1],
				      orientation[2]);
	
	
      }
}

template <int dim>
void Re_MTE_At_UBC<dim>::make_periodicity_constraints (
				  const typename DoFHandler<dim>::face_iterator    &face_1,
				  const typename identity<typename DoFHandler<dim>::face_iterator>::type &face_2,
				  ConstraintMatrix		&constraint_matrix,
				  const ComponentMask		&component_mask,
				  const bool			face_orientation,
				  const bool			face_flip,
				  const bool			face_rotation)
{
// 	    static const int dim = FaceIterator::AccessorType::dimension;

	    Assert( (dim != 1) ||
            (face_orientation == true &&
             face_flip == false &&
             face_rotation == false),
            ExcMessage ("The supplied orientation "
                        "(face_orientation, face_flip, face_rotation) "
                        "is invalid for 1D"));

	    Assert( (dim != 2) ||
            (face_orientation == true &&
             face_rotation == false),
            ExcMessage ("The supplied orientation "
                        "(face_orientation, face_flip, face_rotation) "
                        "is invalid for 2D"));

	    Assert(face_1 != face_2,
	    ExcMessage ("face_1 and face_2 are equal! Cannot constrain DoFs "
                       "on the very same face"));

	    Assert(face_1->at_boundary() && face_2->at_boundary(),
	    ExcMessage ("Faces for periodicity constraints must be on the boundary"));


	    // A lookup table on how to go through the child faces depending on the
	    // orientation:

	    static const int lookup_table_2d[2][2] =
	    {
	      //          flip:
	      {0, 1}, //  false
	      {1, 0}, //  true
	    };

	    static const int lookup_table_3d[2][2][2][4] =
	    {
	      //                    orientation flip  rotation
	      { { {0, 2, 1, 3}, //  false       false false
		  {2, 3, 0, 1}, //  false       false true
		},
		{ {3, 1, 2, 0}, //  false       true  false
		  {1, 0, 3, 2}, //  false       true  true
		},
	      },
	      { { {0, 1, 2, 3}, //  true        false false
		  {1, 3, 0, 2}, //  true        false true
		},
		{ {3, 2, 1, 0}, //  true        true  false
		{2, 0, 3, 1}, //  true        true  true
		},
	      },
	    };

	  // In the case that both faces have children, we loop over all
	  // children and apply make_periodicty_constrains recursively:
	  if (face_1->has_children() && face_2->has_children())
	  {
	    Assert(face_1->n_children() == GeometryInfo<dim>::max_children_per_face &&
		  face_2->n_children() == GeometryInfo<dim>::max_children_per_face,
		  ExcNotImplemented());

	    for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_face; ++i)
	    {
		// Lookup the index for the second face
		unsigned int j;
		switch (dim)
		{
		  case 2:
		  j = lookup_table_2d[face_flip][i];
		  break;
		  case 3:
		  j = lookup_table_3d[face_orientation][face_flip][face_rotation][i];
		  break;
		  default:
		  AssertThrow(false, ExcNotImplemented());
		}

		make_periodicity_constraints (face_1->child(i),
						face_2->child(j),
						constraint_matrix,
						component_mask,
						face_orientation,
						face_flip,
						face_rotation);
	    }
	  }
	else
	  // otherwise at least one of the two faces is active and
	  // we need to enter the constraints
	  {
	    if (face_2->has_children() == false)
	    {
// 	      pcout << "Case 1" << std::endl;
	      set_periodicity_constraints(face_2, face_1,
                                      FullMatrix<double>(IdentityMatrix(face_1->get_fe(face_1->nth_active_fe_index(0)).dofs_per_face)),
                                      constraint_matrix,
                                      component_mask,
                                      face_orientation, face_flip, face_rotation);
	    }
	    else
	    {
// 	      pcout << "Case 2" << std::endl;
	      set_periodicity_constraints(face_1, face_2,
                                      FullMatrix<double>(IdentityMatrix(face_2->get_fe(face_2->nth_active_fe_index(0)).dofs_per_face)),
                                      constraint_matrix,
                                      component_mask,
                                      face_orientation, face_flip, face_rotation);
	    }
	  }
}
template <int dim>
void Re_MTE_At_UBC<dim>::set_periodicity_constraints (
				    const typename DoFHandler<dim>::face_iterator                    	&face_1,
				    const typename identity<typename DoFHandler<dim>::face_iterator>::type	&face_2,
				    const FullMatrix<double>                    				&transformation,
				    ConstraintMatrix                    					&constraint_matrix,
				    const ComponentMask                         				&component_mask,
				    const bool                                   				face_orientation,
				    const bool                                   				face_flip,
				    const bool                                   				face_rotation)
{

//       pcout << "set_periodicity_constraints..1" << std::endl;
//       static const int dim      = FaceIterator::AccessorType::dimension;
      static const int spacedim = dim;

      // we should be in the case where face_1 is active, i.e. has no children:
      Assert (!face_1->has_children(),
              ExcInternalError());

      Assert (face_1->n_active_fe_indices() == 1,
              ExcInternalError());

      // if face_2 does have children, then we need to iterate over them
    if (face_2->has_children())	
    {
//       pcout << "ddd0" << std::endl;
      Assert (face_2->n_children() == GeometryInfo<dim>::max_children_per_face,
	      ExcNotImplemented());
      const unsigned int dofs_per_face
	= face_1->get_fe(face_1->nth_active_fe_index(0)).dofs_per_face;
      FullMatrix<double> child_transformation (dofs_per_face, dofs_per_face);
      FullMatrix<double> subface_interpolation (dofs_per_face, dofs_per_face);
      for (unsigned int c=0; c<face_2->n_children(); ++c)
      {
              // get the interpolation matrix recursively from the one that
              // interpolated from face_1 to face_2 by multiplying from the
              // left with the one that interpolates from face_2 to
              // its child
	  face_1->get_fe(face_1->nth_active_fe_index(0))
	  .get_subface_interpolation_matrix (face_1->get_fe(face_1->nth_active_fe_index(0)),
					      c,
					      subface_interpolation);
	  subface_interpolation.mmult (child_transformation, transformation);
	  set_periodicity_constraints(face_1, face_2->child(c),
				      child_transformation,
				      constraint_matrix, component_mask,
				      face_orientation, face_flip, face_rotation);
	      
      }
//       pcout << "ddd1" << std::endl;
    }
    else
        // both faces are active. we need to match the corresponding DoFs of both faces
    {
//       pcout << "eee0" << std::endl;
      const unsigned int face_1_index = face_1->nth_active_fe_index(0);
      const unsigned int face_2_index = face_2->nth_active_fe_index(0);
          Assert(face_1->get_fe(face_1_index) == face_2->get_fe(face_1_index),
                 ExcMessage ("Matching periodic cells need to use the same finite element"));

      const FiniteElement<dim> &fe = face_1->get_fe(face_1_index);

          Assert(component_mask.represents_n_components(fe.n_components()),
                 ExcMessage ("The number of components in the mask has to be either "
                             "zero or equal to the number of components in the finite " "element."));

      const unsigned int dofs_per_face = fe.dofs_per_face;

      std::vector<types::global_dof_index> dofs_1(dofs_per_face);
      std::vector<types::global_dof_index> dofs_2(dofs_per_face);

      face_1->get_dof_indices(dofs_1, face_1_index);
      face_2->get_dof_indices(dofs_2, face_2_index);

      for (unsigned int i=0; i < dofs_per_face; i++)
      {
	if (dofs_1[i] == numbers::invalid_dof_index ||
	    dofs_2[i] == numbers::invalid_dof_index)
	{
                  /* If either of these faces have no indices, stop.  This is so
                   * that there is no attempt to match artificial cells of
                   * parallel distributed triangulations.
                   *
                   * While it seems like we ought to be able to avoid even calling
                   * set_periodicity_constraints for artificial faces, this
                   * situation can arise when a face that is being made periodic
                   * is only partially touched by the local subdomain.
                   * make_periodicity_constraints will be called recursively even
                   * for the section of the face that is not touched by the local
                   * subdomain.
                   *
                   * Until there is a better way to determine if the cells that
                   * neighbor a face are artificial, we simply test to see if the
                   * face does not have a valid dof initialization.
                   */
	    return;
	}
      }

          // Well, this is a hack:
          //
          // There is no
          //   face_to_face_index(face_index,
          //                      face_orientation,
          //                      face_flip,
          //                      face_rotation)
          // function in FiniteElementData, so we have to use
          //   face_to_cell_index(face_index, face
          //                      face_orientation,
          //                      face_flip,
          //                      face_rotation)
          // But this will give us an index on a cell - something we cannot work
          // with directly. But luckily we can match them back :-]

      std::map<unsigned int, unsigned int> cell_to_rotated_face_index;

          // Build up a cell to face index for face_2:
      for (unsigned int i = 0; i < dofs_per_face; ++i)
      {
	const unsigned int cell_index = fe.face_to_cell_index(i, 0, 
							      /* It doesn't really matter, just assume
                                                            * we're on the first face...
                                                            */
                                                             true, false, false // default orientation
                                                             );
        cell_to_rotated_face_index[cell_index] = i;
      }

//             pcout << "set_periodicity_constraints..2" << std::endl;
          // loop over all dofs on face 2 and constrain them again the ones on face 1
      for (unsigned int i=0; i<dofs_per_face; ++i)
      if (!constraint_matrix.is_constrained(dofs_2[i]))
      if ((component_mask.n_selected_components(fe.n_components())
           == fe.n_components())
           ||
           component_mask[fe.face_system_to_component_index(i).first])
      {
                  // as mentioned in the comment above this function, we need
                  // to be careful about treating identity constraints differently.
                  // consequently, find out whether this dof 'i' will be
                  // identity constrained
                  //
                  // to check whether this is the case, first see whether there are
                  // any weights other than 0 and 1, then in a first stage make sure
                  // that if so there is only one weight equal to 1
	  bool is_identity_constrained = true;
	  for (unsigned int jj=0; jj<dofs_per_face; ++jj)
	  if (((transformation(i,jj) == 0) || (transformation(i,jj) == 1)) == false)
	  {
	    is_identity_constrained = false;
	    break;
	  }
          unsigned int identity_constraint_target = numbers::invalid_unsigned_int;
          if (is_identity_constrained == true)
          {
	    bool one_identity_found = false;
	    for (unsigned int jj=0; jj<dofs_per_face; ++jj)
	    if (transformation(i,jj) == 1)
	    {
	      if (one_identity_found == false)
	      {
		one_identity_found = true;
		identity_constraint_target = jj;
	      }
	      else
	      {
		is_identity_constrained = false;
		identity_constraint_target = numbers::invalid_unsigned_int;
		break;
	      }
	    }
	  }

                  // now treat constraints, either as an equality constraint or
                  // as a sequence of constraints
          if (is_identity_constrained == true)
          {
                      // Query the correct face_index on face_2 respecting the given
                      // orientation:
                      const unsigned int j =
                        cell_to_rotated_face_index[fe.face_to_cell_index(identity_constraint_target,
                                                                         0, /* It doesn't really matter, just assume
                           * we're on the first face...
                           */
                                                                         face_orientation, face_flip, face_rotation)];

                      // if the two aren't already identity constrained (whichever way
                      // around, then enter the constraint. otherwise there is nothing
                      // for us still to do
									 
										    
// 			  const bool ddd = constraint_velocity.are_identity_constrained(dofs_2[i], dofs_1[i]);
// 			  const ConstraintLine &p = lines[lines_cache[calculate_line_index(dofs_2[i])]];
// 									 bool dd = constraint_matrix.is_identity_constrained(dofs_2[i]);
// 			  const ConstraintMatrix::ConstraintLine p = constraint_matrix.lines[constraint_matrix.lines_cache[constraint_matrix.calculate_line_index(dofs_2[i])]];
// 			  ConstraintMatrix::ConstraintLine p;
// 			  unsigned int ddd = p.calculate_line_index(dofs_2[i]);
// 			  unsigned int ddd = constraint_matrix.calculate_line_index(dofs_2[i]);
			
//                       if (constraint_matrix.are_identity_constrained(dofs_2[i], dofs_1[i]) == false)
			 if (constraint_matrix.is_identity_constrained(dofs_2[i]) == false &&
			   constraint_matrix.is_identity_constrained(dofs_1[i]) == false)
                        {
                          constraint_matrix.add_line(dofs_2[i]);
                          constraint_matrix.add_entry(dofs_2[i], dofs_1[j], 1);
                        }
          }
          else
          {
                      // this is just a regular constraint. enter it piece by piece
                      constraint_matrix.add_line(dofs_2[i]);
                      for (unsigned int jj=0; jj<dofs_per_face; ++jj)
                        {
                          // Query the correct face_index on face_2 respecting the given
                          // orientation:
                          const unsigned int j =
                            cell_to_rotated_face_index[fe.face_to_cell_index(jj, 0, 
				/* It doesn't really matter, just assume
                               * we're on the first face...*/
                                                                             face_orientation, face_flip, face_rotation)];

                          // And finally constrain the two DoFs respecting component_mask:
                          if (transformation(i,jj) != 0)
                            constraint_matrix.add_entry(dofs_2[i], dofs_1[j],
                                                        transformation(i,jj));
                        }
           }
      } //loop_constraint_matrix
// 	  pcout << "eee1" << std::endl;
    } //if(face_2->child?)
}  
  
template <int dim>
void Re_MTE_At_UBC<dim>::refine_mesh (bool is_init_matrix)
{
    pcout <<"* Refinement.. " << is_init_matrix << std::endl;
    if (is_solid_particle == true) particle_refine_mesh ();

// if (is_mass_trn == true)
// {
	   // need to do later for mass transfer
// }

    prescribe_flag_refinement ();
    adp_execute_transfer (is_init_matrix);
}

template <int dim>
void Re_MTE_At_UBC<dim>::particle_refine_mesh ()
{
  //ParsRef
    if (is_verbal_output == true) 
      pcout << "* Particle Refine Mesh..." << std::endl;
    typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

    unsigned int iij = 0;
    for (; cell!=endc; ++cell, ++iij)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
	ParsRef [iij] = 0;
	unsigned int cell_level = cell->level();

        Point<dim> c = cell->center();
        std::pair<unsigned int,double> distant_of_par = distant_from_particles (c);
        double min_tq = std::abs (distant_of_par.second);

        double aa = 0;
        double bb = 0;
        for (unsigned int i=0; i<reLev; ++i)
        {
            aa = bb;
            bb = aa + 2*i;
            if (i==0) bb = 1;

            if (min_tq > aa*safe_fac && min_tq <= bb*safe_fac)
            {
                if (cell_level < max_level-i) ParsRef [iij] = 1;
                if (cell_level > max_level-i) ParsRef [iij] = -1;
                if (cell_level == max_level-i)
                {
                    ParsRef [iij] = 0;
                }
            }
        }

        if (min_tq > bb*safe_fac)
            ParsRef [iij] = -1;
	
	for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
            const Point<dim> face_center = cell->face(f)->center();

            if (cell->face(f)->at_boundary())
            {
		if (std::abs(face_center[0]-(xmin)) < 1e-6)
		{
// 		    ParsRef [iij] = +1;
		}
		
		if (std::abs(face_center[0]-(xmax)) < 1e-6)
		{
// 		    ParsRef [iij] = +1;
		}
	    }
	}
    }
}

template <int dim>
void Re_MTE_At_UBC<dim>::prescribe_flag_refinement ()
{
    pcout << "* Prescribe Ref. Flag..." << std::endl;

    unsigned int iij = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell=triangulation.begin_active();
         cell!=triangulation.end(); ++cell, ++iij)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
        cell->clear_coarsen_flag (); cell->clear_refine_flag ();

	if (ParsRef[iij] == +1) cell->set_refine_flag ();
	if (ParsRef[iij] == -1) cell->set_coarsen_flag ();
        
	unsigned int cell_level = static_cast<unsigned int> (cell->level());
	
	if (cell_level == min_level) cell->clear_coarsen_flag ();
	if (cell_level == max_level) cell->clear_refine_flag ();
    }
}

template <int dim>
void Re_MTE_At_UBC<dim>::adp_execute_transfer (bool is_init_matrix)
{

//    TrilinosWrappers::Vector		vel_star;
//    TrilinosWrappers::Vector		vel_n_plus_1, vel_n, vel_n_minus_1;
//    TrilinosWrappers::Vector		vel_nPlus_con, vel_n_con;
//    TrilinosWrappers::Vector		pre_n_plus_1, pre_n;
//    TrilinosWrappers::Vector		aux_n_plus_1, aux_n, aux_n_minus_1;
    
    if (is_verbal_output == true) 
      pcout << "* Adp. Exeu. Trans.. UPL " << std::endl;

    std::vector<TrilinosWrappers::Vector> x_vel (4), x_pre (2), x_aux (3), x_vel_con (2);

    x_vel [0] = vel_n_plus_1;
    x_vel [1] = vel_n;
    x_vel [2] = vel_n_minus_1;
    x_vel [3] = vel_star;

    x_pre [0] = pre_n_plus_1;
    x_pre [1] = pre_n;

    x_aux [0] = aux_n_plus_1;
    x_aux [1] = aux_n;
    x_aux [2] = aux_n_minus_1;

    x_vel_con [0] = vel_nPlus_con;
    x_vel_con [1] = vel_n_con;
    
    SolutionTransfer<dim,TrilinosWrappers::Vector> vel_trans (dof_handler_velocity);
    SolutionTransfer<dim,TrilinosWrappers::Vector> pre_trans (dof_handler_pressure);
    SolutionTransfer<dim,TrilinosWrappers::Vector> aux_trans (dof_handler_pressure);
    SolutionTransfer<dim,TrilinosWrappers::Vector> vel_on_concentr_trans (dof_handler_velocity_on_concentr);

    triangulation.prepare_coarsening_and_refinement();

    vel_trans.prepare_for_coarsening_and_refinement(x_vel);
    pre_trans.prepare_for_coarsening_and_refinement(x_pre);
    aux_trans.prepare_for_coarsening_and_refinement(x_aux);
    vel_on_concentr_trans.prepare_for_coarsening_and_refinement(x_vel_con);

    triangulation.execute_coarsening_and_refinement ();

    setup_dofs (true, true, is_init_matrix);
//     setup_dofs_con (true, is_init_matrix);
    
    std::vector<TrilinosWrappers::Vector> tmp_vel (4), tmp_pre (2), tmp_aux (3), tmp_vel_on_concentr (2);

    tmp_vel[0].reinit (vel_n_plus_1);
    tmp_vel[1].reinit (vel_n);
    tmp_vel[2].reinit (vel_n_minus_1);
    tmp_vel[3].reinit (vel_star);

    tmp_pre[0].reinit (pre_n_plus_1);
    tmp_pre[1].reinit (pre_n);

    tmp_aux[0].reinit (aux_n_plus_1);
    tmp_aux[1].reinit (aux_n);
    tmp_aux[2].reinit (aux_n_minus_1);

    tmp_vel_on_concentr[0].reinit (vel_nPlus_con);
    tmp_vel_on_concentr[1].reinit (vel_n_con);

    vel_trans.interpolate (x_vel, tmp_vel);
    pre_trans.interpolate (x_pre, tmp_pre);
    aux_trans.interpolate (x_aux, tmp_aux);
    vel_on_concentr_trans.interpolate (x_vel_con, tmp_vel_on_concentr);

    vel_n_plus_1 = tmp_vel[0];
    vel_n = tmp_vel[1];
    vel_n_minus_1 = tmp_vel[2];
    vel_star = tmp_vel[3];

    pre_n_plus_1 = tmp_pre[0];
    pre_n = tmp_pre[1];

    aux_n_plus_1 = tmp_aux[0];
    aux_n = tmp_aux[1];
    aux_n_minus_1 = tmp_aux[2];

    vel_nPlus_con = tmp_vel_on_concentr[0];
    vel_n_con = tmp_vel_on_concentr[1];  
}

template <int dim>
void Re_MTE_At_UBC<dim>::extrapolation_step ()
{
    if (is_verbal_output == true) 
      pcout << "  Extrpolation Step..  ";
    
    vel_star = 0;
	  
    vel_star.equ (2.0, vel_n, -1.0, vel_n_minus_1);
    
    pcout << std::endl;
}

template <int dim>
void Re_MTE_At_UBC<dim>::diffusion_step ()
{
    if (is_verbal_output == true) pcout << "  Diffusion Step..  ";

    matrix_velocity = 0;
    rhs_velocity = 0;

    double inv_time_step = 1./time_step;

    const QGauss<dim> quadrature_formula((fe_pressure.get_degree() + 1)+1);

    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);
    
    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);
    
    const unsigned int dofs_per_cell = fe_velocity.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<double>  pre_star_values (n_q_points);
    
    std::vector<double>  viscosity_values (n_q_points);
    
    std::vector<Tensor<1,dim> > vel_star_values (n_q_points);
    std::vector<Tensor<1,dim> > vel_n_values (n_q_points);
    std::vector<Tensor<1,dim> > vel_n_minus_1_values (n_q_points);
//     std::vector<std::vector<Tensor<1,dim> > >
//         grad_vel_star_values (n_q_points, std::vector<Tensor<1,dim> >(dim));
    std::vector<Tensor<2,dim> > grad_vel_star_values (n_q_points);
    

    std::vector<Tensor<1,dim> > grad_aux_n_values (n_q_points);
    std::vector<Tensor<1,dim> > grad_aux_n_minus_1_values (n_q_points);

    std::vector<Tensor<1,dim> > grad_pre_n_values (n_q_points);

    std::vector<Tensor<1,dim> >          phi_u (dofs_per_cell);
    std::vector<Tensor<2,dim> >          grads_phi_u (dofs_per_cell);
    std::vector<SymmetricTensor<2,dim> > symm_grads_phi_u (dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler_velocity.begin_active(),
	endc = dof_handler_velocity.end(),
	pre_cell = dof_handler_pressure.begin_active();
	
    for (; cell!=endc; ++cell, ++pre_cell)
        if (cell->subdomain_id() == 
	  Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
        {
            fe_values_velocity.reinit (cell);
            fe_values_pressure.reinit (pre_cell);

            cell->get_dof_indices (local_dofs_indices);

            fe_values_velocity[velocities].get_function_values (vel_star, vel_star_values);
            fe_values_velocity[velocities].get_function_values (vel_n_minus_1, vel_n_minus_1_values);
            fe_values_velocity[velocities].get_function_values (vel_n, vel_n_values);
            fe_values_velocity[velocities].get_function_gradients (vel_star, grad_vel_star_values);

	    fe_values_pressure.get_function_gradients (aux_n, grad_aux_n_values);
	    fe_values_pressure.get_function_gradients (aux_n_minus_1, grad_aux_n_minus_1_values);
	    fe_values_pressure.get_function_gradients (pre_n, grad_pre_n_values);
// 	    fe_values_pressure.get_function_values (particle_distb, viscosity_values);
	    
            local_matrix = 0;
            local_rhs = 0;
	    
	    double nu = 1.0;
	    
// 	    if (turn_on_periodic_bnd)
	    {
	      Point<dim> c = cell->center();
	      std::pair<unsigned int,double> distant_of_par = distant_from_particles (c);
	      if (distant_of_par.second <0) nu = FacPar;
	    }
	    
	    
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
		      phi_u[k] = fe_values_velocity[velocities].value (k,q);
		      grads_phi_u[k] = fe_values_velocity[velocities].gradient (k,q);
		      symm_grads_phi_u[k] = fe_values_velocity[velocities].symmetric_gradient(k,q);
                }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
		    //Time-stepping
		    local_rhs(i) -=	Reynolds_number*
					phi_u[i]*
					(
					  -2.0*
					  vel_n_values[q]
					  +
					  0.5*
					  vel_n_minus_1_values[q]
					)*
					fe_values_velocity.JxW(q);

		    //Gradient of pressure
		    local_rhs(i) -= 	time_step*
					phi_u[i]*
					(
					    grad_pre_n_values[q] +
					    (4./3.)*grad_aux_n_values[q] -
					    (1./3.)*grad_aux_n_minus_1_values[q]
					)*
					fe_values_velocity.JxW(q);

                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        const unsigned int comp_j = fe_velocity.system_to_component_index (j).first;

			//Time-stepping
                        local_matrix(i,j) +=    Reynolds_number*
						 1.5*
                                                phi_u[i]*
                                                phi_u[j]*
                                                fe_values_velocity.JxW(q);

			//Advection Term
			{
			    local_matrix(i,j) +=	Reynolds_number*
							time_step*
							grads_phi_u[j]*
							vel_star_values[q]*
							phi_u[i]*
							fe_values_velocity.JxW(q);

			    local_matrix(i,j) +=	Reynolds_number*
							time_step*
							0.5*
							phi_u[i]*
							(grad_vel_star_values[q][0][0]
							+
							grad_vel_star_values[q][1][1])*
							phi_u[j]*
							fe_values_velocity.JxW(q);

			}

			//Viscous term
                        local_matrix(i,j) +=    time_step*
                                                (2.0)*
                                                nu*
                                                symm_grads_phi_u[i]*
                                                symm_grads_phi_u[j]*
                                                fe_values_velocity.JxW(q);


                    }
                }
            }
            constraint_velocity.distribute_local_to_global (local_matrix,
                                                            local_dofs_indices,
                                                            matrix_velocity);

            constraint_velocity.distribute_local_to_global (local_rhs,
                                                            local_dofs_indices,
                                                            rhs_velocity);
        }

    std::map<unsigned int,double> boundary_values;

    std::vector<bool> vel_prof(dim, true);

    unsigned int n_u = dof_handler_velocity.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association 
			      (dof_handler_velocity,
			      Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

    TrilinosWrappers::MPI::Vector distributed_vel_n_plus_1 (map);

    {
	std::vector<bool> vel_prof(dim, true);
	MappingQ<dim> ff (fe_pressure.get_degree()+1);
	VectorTools::interpolate_boundary_values (ff,
						dof_handler_velocity,
						0,
						ZeroFunction<dim>(dim),
						boundary_values,
						vel_prof);

	VectorTools::interpolate_boundary_values (ff,
						dof_handler_velocity,
						6,
						Inflow_Velocity<dim>(mean_velocity, Inlet_velocity_type),
						boundary_values,
						vel_prof);
	
	VectorTools::interpolate_boundary_values (ff,
						dof_handler_velocity,
						7,
						Inflow_Velocity<dim>(mean_velocity, Inlet_velocity_type),
						boundary_values,
						vel_prof);
	
	if (turn_on_periodic_bnd == false)
	VectorTools::interpolate_boundary_values (ff,
						dof_handler_velocity,
						3,
						Inflow_Velocity<dim>(mean_velocity, Inlet_velocity_type),
						boundary_values,
						vel_prof);

	if (Inlet_velocity_type == 1)
	{
	  std::vector<bool> vel_prof2(dim, true);
	  if (dim == 2) vel_prof2[0] = false;
	  if (dim == 3) vel_prof2[2] = false;

	  if (turn_on_periodic_bnd == false)
	  VectorTools::interpolate_boundary_values (ff,
						  dof_handler_velocity,
						  4,
						  ConstantFunction<dim>(0.0),
						  boundary_values,
						  vel_prof2);
	}
	
	MatrixTools::apply_boundary_values (boundary_values,
					    matrix_velocity,
					    distributed_vel_n_plus_1,
					    rhs_velocity,
					    false);	
    }

    matrix_velocity.compress(VectorOperation::add);
    rhs_velocity.compress(VectorOperation::add);

    distributed_vel_n_plus_1 = vel_n_plus_1;
    SolverControl solver_control (matrix_velocity.m(), error_crit_NS*rhs_velocity.l2_norm ());

    SolverGMRES<TrilinosWrappers::MPI::Vector>
    gmres (solver_control,SolverGMRES<TrilinosWrappers::MPI::Vector >::AdditionalData(100));

    TrilinosWrappers::PreconditionILU preconditioner;
    preconditioner.initialize (matrix_velocity);
    gmres.solve (matrix_velocity, distributed_vel_n_plus_1, rhs_velocity, preconditioner);

    vel_n_plus_1 = distributed_vel_n_plus_1;

    if (is_verbal_output == true)
    pcout	<< solver_control.last_step()
		<< std::endl;

    constraint_velocity.distribute (vel_n_plus_1);
    
}

template <int dim>
void Re_MTE_At_UBC<dim>::projection_step ()
{
    if (is_verbal_output == true) 
      pcout << "  Projection Step.. ";

    matrix_pressure = 0;
    
    rhs_pressure = 0;

    double inv_time_step = 1./time_step;

    double kappa = 1.0;
    
    const QGauss<dim> quadrature_formula (fe_pressure.get_degree()+1);

    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);

    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
                                      update_values |
                                      update_gradients);

    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);

    Vector<double> local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<std::vector<Tensor<1,dim> > >
	grad_vel_n_plus_1_values (n_q_points, std::vector<Tensor<1,dim> >(dim));

    typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler_pressure.begin_active(),
	endc = dof_handler_pressure.end(),
	vel_cell = dof_handler_velocity.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell)
        if (cell->subdomain_id() == 
	  Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
        {
            fe_values_pressure.reinit (cell);
            fe_values_velocity.reinit (vel_cell);

            fe_values_velocity.get_function_gradients(vel_n_plus_1, grad_vel_n_plus_1_values);

            cell->get_dof_indices (local_dofs_indices);

            local_matrix = 0;
            local_rhs = 0;
		      
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
		    local_matrix(i,j) -= (1./kappa)*
					  time_step*
					  0.75*
					  fe_values_pressure.shape_grad(i, q)*
					  fe_values_pressure.shape_grad(j, q)*
					  fe_values_pressure.JxW(q);

                for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
		      double bb = 0.0;
		      for (unsigned int d=0; d<dim; ++d)
			bb += grad_vel_n_plus_1_values[q][d][d];
		      
		      local_rhs(i) += 	fe_values_pressure.shape_value(i, q)*
					bb*
					fe_values_pressure.JxW(q);
				    
		  }
            }

            constraint_pressure.distribute_local_to_global (local_matrix,
                                                            local_dofs_indices,
                                                            matrix_pressure);

            constraint_pressure.distribute_local_to_global (local_rhs,
                                                            local_dofs_indices,
                                                            rhs_pressure);
        }

    unsigned int n_p = dof_handler_pressure.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association 
				(dof_handler_pressure,
				Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));

    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

    TrilinosWrappers::MPI::Vector distibuted_aux_n_plus_1 (map);
    
    std::map<unsigned int,double> boundary_values;

//     unsigned int impose_press_bnd_type = 0;
    
//     if (	(xmax_bnd == 4 && dim == 2) || 
// 		(zmax_bnd == 4 && dim == 3)  )
//     if (Inlet_velocity_type == 1)
//       impose_press_bnd_type = 1;
    
//     if (impose_press_bnd_type == 1) 
//     {
//       MappingQ<dim> ff (fe_pressure.get_degree());
//       VectorTools::interpolate_boundary_values (ff,
// 						dof_handler_pressure,
// 						4,
// 						ConstantFunction<dim> (0.0),
// 						boundary_values);
//     } 
//     else if (impose_press_bnd_type == 0) 
    {
      typename DoFHandler<dim>::active_cell_iterator  
	initial_cell = dof_handler_pressure.begin_active();
      unsigned int dof_number_pressure = initial_cell->vertex_dof_index(0 , 0);
      boundary_values[dof_number_pressure] = 0.0;
    }
    
    MatrixTools::apply_boundary_values (boundary_values,
                                        matrix_pressure,
                                        distibuted_aux_n_plus_1,
                                        rhs_pressure,
                                        false);

    matrix_pressure.compress(VectorOperation::add);
    rhs_pressure.compress(VectorOperation::add);

    distibuted_aux_n_plus_1 = aux_n_plus_1;
    SolverControl solver_control (matrix_pressure.m(), error_crit_NS*rhs_pressure.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize (matrix_pressure);
    cg.solve (matrix_pressure, distibuted_aux_n_plus_1, rhs_pressure, preconditioner);
    aux_n_plus_1 = distibuted_aux_n_plus_1;

//     if (is_verbal_output == true)
      pcout	<< solver_control.last_step()
		<< std::endl;

    constraint_pressure.distribute (aux_n_plus_1);
}

	template <int dim>
	void Re_MTE_At_UBC<dim>::pressure_correction_step_stn ()
	{

	//     if (is_verbal_output == true)
	//       pcout << "  Correction Step.. Stn   " << std::endl;

	    pre_n_plus_1.equ(1.0, pre_n , 1.0, aux_n_plus_1);

	}
	
template <int dim>
void Re_MTE_At_UBC<dim>::pressure_correction_step_rot ()
{
    if (is_verbal_output == true)
      pcout << "  Correction Step.. Rot   ";

    matrix_pressure = 0;
    rhs_pressure = 0;

    double inv_time_step = 1./time_step;

    const QGauss<dim> quadrature_formula (fe_pressure.get_degree()+1);

    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);

    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
                                      update_values |
                                      update_gradients);

    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);

    Vector<double> local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<Tensor<2,dim> > grad_vel_sol_values (n_q_points);
    std::vector<Tensor<1,dim> > vel_sol_values (n_q_points);
    std::vector<double>	aux_sol_values (n_q_points);
    std::vector<double>	pre_sol_values (n_q_points);
    std::vector<double>	viscosity_values (n_q_points);
    
    const FEValuesExtractors::Vector velocities (0);
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_pressure.begin_active(),
    endc = dof_handler_pressure.end();

    typename DoFHandler<dim>::active_cell_iterator
    vel_cell = dof_handler_velocity.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell)
        if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
        {
	    cell->get_dof_indices (local_dofs_indices);

            fe_values_pressure.reinit (cell);
            fe_values_velocity.reinit (vel_cell);

	    fe_values_velocity[velocities].get_function_gradients (vel_n_plus_1, grad_vel_sol_values);
	    fe_values_velocity[velocities].get_function_values (vel_n_plus_1, vel_sol_values);
	    fe_values_pressure.get_function_values (aux_n_plus_1, aux_sol_values);
	    fe_values_pressure.get_function_values (pre_n, pre_sol_values);
// 	    fe_values_pressure.get_function_values (particle_distb, viscosity_values);
	    
            local_matrix = 0;
            local_rhs = 0;

	    double nu = 1.0;
	    
	    if (turn_on_periodic_bnd)
	    {
	      Point<dim> c = cell->center();
	      std::pair<unsigned int,double> distant_of_par = distant_from_particles (c);
	      if (distant_of_par.second <0) nu = FacPar;
	    }
	    
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    local_matrix(i,j)   +=  fe_values_pressure.shape_value(i, q)*
                                            fe_values_pressure.shape_value(j, q)*
                                            fe_values_pressure.JxW(q);

                }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    local_rhs(i) += fe_values_pressure.shape_value(i, q)*
				     (
					pre_sol_values[q] +
					aux_sol_values[q]
                                    )*
                                    fe_values_pressure.JxW(q);
				      
// 		    double bb = 0.0;
// 		    for (unsigned int d=0; d<dim; ++d)
// 		      bb += grad_vel_sol_values[q][d][d];
		    
//                     local_rhs(i) -=	fe_values_pressure.shape_value(i, q)*
// 					bb*
// 					nu*
// // 					viscosity_values[q]*
// 					(1./Reynolds_number)*
// 					fe_values_pressure.JxW(q);
                }
            }

            constraint_pressure.distribute_local_to_global (local_matrix,
                                                            local_dofs_indices,
                                                            matrix_pressure);

            constraint_pressure.distribute_local_to_global (local_rhs,
                                                            local_dofs_indices,
                                                            rhs_pressure);
        }

    unsigned int n_p = dof_handler_pressure.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association 
				(dof_handler_pressure,
				Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));

    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::MPI::Vector distibuted_pre_sol (map);
    
    std::map<unsigned int,double> boundary_values;

    unsigned int impose_press_bnd_type = 0;
    
    if (	(xmax_bnd == 4 && dim == 2) || 
		(zmax_bnd == 4 && dim == 3)  )
    if (Inlet_velocity_type == 1)
      impose_press_bnd_type = 1;
    
    if (impose_press_bnd_type == 1) 
    {
      MappingQ<dim> ff (fe_pressure.get_degree());
      VectorTools::interpolate_boundary_values (ff,
						dof_handler_pressure,
						4,
						ConstantFunction<dim> (0.0),
						boundary_values);
    } 
    else if (impose_press_bnd_type == 0) 
    {
      typename DoFHandler<dim>::active_cell_iterator  initial_cell = dof_handler_pressure.begin_active();
      unsigned int dof_number_pressure = initial_cell->vertex_dof_index(0 , 0);
      boundary_values[dof_number_pressure] = 0.0;
    }
    
//     MatrixTools::apply_boundary_values (boundary_values,
//                                         matrix_pressure,
//                                         distibuted_pre_sol,
//                                         rhs_pressure,
//                                         false);
    
    matrix_pressure.compress(VectorOperation::add);
    rhs_pressure.compress(VectorOperation::add);

    SolverControl solver_control (matrix_pressure.m(), error_crit_NS*rhs_pressure.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> cg (solver_control);

    //TrilinosWrappers::PreconditionILU preconditioner;
    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize (matrix_pressure);
    cg.solve (matrix_pressure, distibuted_pre_sol, rhs_pressure, preconditioner);
    pre_n_plus_1 = distibuted_pre_sol;

//     if (is_verbal_output == true)
      pcout   << solver_control.last_step()
	      << std::endl;

    constraint_pressure.distribute (pre_n_plus_1);      
}

template <int dim>
void Re_MTE_At_UBC<dim>::solution_update ()
{
    if (is_verbal_output == true)
      pcout << "  Solution Update Step.. " << std::endl;

    vel_n_minus_1 = vel_n;
    vel_n = vel_n_plus_1;

    pre_n = pre_n_plus_1;

    aux_n_minus_1 = aux_n;
    aux_n = aux_n_plus_1;

}

template <int dim>
void Re_MTE_At_UBC<dim>::vel_pre_convergence ()
{
    vel_res = vel_n_plus_1;
    pre_res = pre_n_plus_1;

    vel_res.sadd (1, -1, vel_n);
    pre_res.sadd (1, -1, pre_n);
    
    pressure_l2_norm = pre_res.l2_norm();
    
    pcout   << "  "
	    << vel_res.l2_norm()
	    << ", "
	    << pre_res.l2_norm()
	    << std::endl;
}

template <int dim>
void Re_MTE_At_UBC<dim>::transfer_velocities_on_concentr ()
{
    pcout << "# Transfer Vel On Concentr..." << std::endl;
    
    Vector<double> a (vel_n); a = vel_n;
    Vector<double> tmp1 (vel_nPlus_con);
    VectorTools::interpolate_to_different_mesh (dof_handler_velocity,
						  a,
						  dof_handler_velocity_on_concentr,
						  tmp1);
    vel_nPlus_con = tmp1; tmp1 = 0; a = vel_n_minus_1;
    VectorTools::interpolate_to_different_mesh (dof_handler_velocity,
						  a,
						  dof_handler_velocity_on_concentr,
						  tmp1);
    vel_n_con = tmp1;
}

template <int dim>
void Re_MTE_At_UBC<dim>::assigning_tracer_distribution ()
{
  double x_length = std::abs(xmax-xmin);
  
    for (unsigned int i=0; i<num_of_tracers; ++i)
    {
      Point<dim> aa;
	
      if (no_assign_type == 0)
      {
	double dd = static_cast<double> (x_length/(num_of_tracers));
	
	aa[0] = xmin + double(i)*dd;
	aa[1] = threshold_ypos;
      }
      
      if (no_assign_type == 1)
      {
	double dd = static_cast<double> (x_length/(0.5*num_of_tracers));
	
	double sign_val = +1;
	if (i%2) sign_val = -1;
	
	aa[0] = xmin + double(i)*dd*0.5;
	aa[1] = sign_val*threshold_ypos;
      }
   
      if (no_assign_type == 2)
      {
	std::vector<unsigned int> dum;
        dum.push_back (int(1)*10000);
        dum.push_back (int(1)*10000);
	
	bool valid_gen_tmpPar = true;
	do 
	{
	  valid_gen_tmpPar = true;
	  
	  for (unsigned int d = 0; d < dim ; ++d)
	  {
	    unsigned int num0 = rand()%dum[d];
	    double num1 = num0;
	    double leg_max = 0.0;
            if (d == 0) leg_max = xmax;
            if (d == 1) leg_max = ymax;
            num1 = num1/10000 - leg_max;
            aa[d] = num1;
	  }
	  
	  for (unsigned int n=0; n<num_pars; ++n)
	  {
	    std::pair<unsigned int,double> distant_of_par = distant_from_particles (aa);
	    if (distant_of_par.second <0) 
	    {
	      valid_gen_tmpPar = false;
	    }
	  }
	  
	 
	} while (valid_gen_tmpPar == false);
		
	aa[0] = aa[0];
	aa[1] = aa[1];
      }
    
      unsigned int which_type_of_tracer = 0;
    
      tracer_position.push_back (aa);
      old_tracer_position.push_back (aa);
      what_type_tracer.push_back (which_type_of_tracer);
      if (aa[0] > 0.5*(xmax+xmin)) 
	aa[0] = aa[0] - x_length;
      else aa[0] = aa[0] + x_length;
      image_tracer_position.push_back (aa);
    
      Point<dim> dum; 
      tracer_velocity.push_back(dum); old_tracer_velocity.push_back(dum);
      fluid_velocity_at_tracer.push_back(dum);
    }
    
    for (unsigned int i=0; i<num_of_tracers; ++i)
    {  
      unsigned int which_type_of_tracer = 0;
      if (i < 0.5*num_of_tracers)
      {	which_type_of_tracer = 1; what_type_tracer[i] = which_type_of_tracer;}
      if (i >= 0.5*num_of_tracers)
      {	which_type_of_tracer = 2; what_type_tracer[i] = which_type_of_tracer;}
    }
}

template <int dim>
void Re_MTE_At_UBC<dim>::find_fluid_veloicity_at_tracer_position ()
{
    pcout << "* Find Fluid Vel at Tracers" << std::endl;

    Vector<double> aa(dim);
    for (unsigned int i=0; i<tracer_position.size(); ++i)
    {
      Vector<double> bb = vel_n_plus_1;
      Point<dim> cc = tracer_position[i];

      MappingQ<dim> ff((fe_pressure.get_degree() + 1));
      VectorTools::point_value (ff,
				 dof_handler_velocity,
				 bb,
				 cc,
				 aa);

      for(unsigned int d=0; d<dim; ++d)
	fluid_velocity_at_tracer[i][d] = aa[d];
    }
}

template <int dim>
void Re_MTE_At_UBC<dim>::advancement_for_tracer ()
{
    pcout << "* Advancement Tracers..." << std::endl;
    double x_length = std::abs(xmax-xmin);
    
    double diffusivity = 1./Peclet_number;
    for (unsigned int i=0; i<tracer_position.size(); ++i)
    {
      Point<dim> bb;
      
      for (unsigned int d = 0; d < dim ; ++d)
      {
	unsigned int num0 = rand()%10000;
	double num1 = static_cast<double> (num0);
	num1 = num1/10000;
	num1 = 2.*(num1-0.5);
	
        bb[d] =	 num1*
		 std::sqrt(6.0*diffusivity*time_step);
      
      }
      old_tracer_position[i] = tracer_position[i];
      if (Peclet_number < 1e+8) tracer_position[i] +=  time_step*fluid_velocity_at_tracer[i] + bb;
      if (Peclet_number > 1e+8) tracer_position[i] +=  time_step*fluid_velocity_at_tracer[i];
      
      //Wall or/and Particle surface reflecting
      {
	  bool do_bisection_search_particle = false;
	  for (unsigned int n=0; n<num_pars; ++n)
	  {
	    double aa0 = tracer_position[i].distance(cenPar[n]) - par_rad;
	    if (aa0 < 0.0) do_bisection_search_particle = true; 
	  }
	
	  if (do_bisection_search_particle == true && FacPar > 1.0)
	  {
	    Point<dim> new_adjust_pos;
	    bi_section_search (  old_tracer_position[i], 
				  tracer_position[i],
				  cenPar,
				  new_adjust_pos);
	  
// 	  	tracer_position[i] = 0.5*(new_adjust_pos + old_tracer_position[i]);
// 	  	tracer_position[i] = old_tracer_position[i];
	  
	    Point<dim> normal_vector;
	    normal_vector = old_tracer_position[i] - new_adjust_pos;
	    normal_vector = normal_vector/normal_vector.norm();
	  
	    double dist_xx = new_adjust_pos.distance(tracer_position[i]);
	    tracer_position[i] = new_adjust_pos + normal_vector*(dist_xx+1e-2);

	  }
	
	  //Wall Bounding
	  if (tracer_position[i][1] < ymin)
	  {
	    tracer_position[i][1] = ymin + std::abs(tracer_position[i][1] - ymin);
	    if (what_type_tracer[i] == 1)
	    {
	      what_type_tracer[i] = 3;
	      touch_at_ymin = touch_at_ymin + 1.0;
	    }
	  }
	  if (tracer_position[i][1] > ymax)
	  {
	    tracer_position[i][1] = ymax - std::abs(tracer_position[i][1] - ymax);
	    if (what_type_tracer[i] == 2)
	    {
	      what_type_tracer[i] = 3;
	      touch_at_ymax = touch_at_ymax + 1.0;
	    }	    
	  }
	}	
      
	if (tracer_position[i][0] > 0.5*(xmax+xmin)) 
	  image_tracer_position[i][0] = tracer_position[i][0] - x_length;
	else image_tracer_position[i][0] = tracer_position[i][0] + x_length;
    
	image_tracer_position [i][1] = tracer_position [i][1];
      
	Point<dim> dummy;
	if (tracer_position [i][0] < xmin || tracer_position [i][0] > xmax)
	{
	  dummy = tracer_position [i];
	  tracer_position [i] = image_tracer_position[i];
	  image_tracer_position[i] = dummy;
	}
    }
}

template <int dim>
void Re_MTE_At_UBC<dim>::bi_section_search (	Point<dim> &old_tracer, 
						Point<dim> &current_tracer,
						std::vector<Point<dim> > &particle,
						Point<dim> &new_adjust_pos)
{
//   pcout << "*** Bi-section..." << std::endl;
  bool continue_routie = true;
  Point<dim> left_point, right_point;
  for (unsigned int d=0; d<dim; ++d)
  {
    left_point[d] = old_tracer[d]; 
    right_point[d] = current_tracer[d];
  }
		
  unsigned int counter = 0;
  do 
  {
    continue_routie = false;
    
    for (unsigned int d=0; d<dim; ++d)
      new_adjust_pos[d] = 0.5*(left_point[d] + right_point[d]);
    
    double small_dist_from_particle = 1000.0;
    unsigned int which_particle = 10000;
    
    for (unsigned int n=0; n<num_pars; ++n)
    {
      double dd = particle[n].distance(new_adjust_pos) - par_rad;
      
      if (std::abs(dd) < small_dist_from_particle) 
      {
	which_particle = n;
	small_dist_from_particle = dd;
      }
    }

    
    Point<dim> mm;
    for (unsigned int d=0; d<dim; ++d)
      mm[d] = std::abs(left_point[d] - right_point[d]);
      
//     if (small_dist_from_particle > 0.0)
//     if (std::abs(small_dist_from_particle) < 1e-3 || mm.norm() < 1e-6)
      if (mm.norm() < 1e-11)
	continue_routie = true;
    
    
//     if (counter < 100)	
//     pcout	<< left_point << " | "
// 		<< right_point << " | "
// 		<< small_dist_from_particle << " | " 
// 		<< mm.norm() << std::endl;
		
    if (continue_routie == false)
    {
      if (small_dist_from_particle > 0.0)
      {
	for (unsigned int d=0; d<dim; ++d)
	  right_point[d] = new_adjust_pos[d];
      }
      
      if (small_dist_from_particle < 0.0)
      {
	for (unsigned int d=0; d<dim; ++d)
	  left_point[d] = new_adjust_pos[d];
      }
    }
    ++counter;
  } while (continue_routie == false);
}

template <int dim>
void Re_MTE_At_UBC<dim>::assemble_matrix_for_mass ()
{
    pcout << "* Assemble system for C..." << std::endl;

    const bool use_bdf2_scheme = (timestep_number != 0);
    matrix_concentr = 0;

    const QGauss<dim> quadrature_formula (fe_concentr.get_degree() +1 );

    FEValues<dim>     fe_concentr_values (fe_concentr, quadrature_formula,
                                               update_values    |
                                               update_gradients |
                                               update_hessians  |
                                               update_quadrature_points  |
                                               update_JxW_values);

    const unsigned int   dofs_per_cell   = fe_concentr.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   local_mass_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   local_stiffness_matrix (dofs_per_cell, dofs_per_cell);
    
    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);
    std::vector<double>         phi_T      (dofs_per_cell);
    std::vector<Tensor<1,dim> > grad_phi_T (dofs_per_cell);
    
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_concentr.begin_active(),
      endc = dof_handler_concentr.end();

    double DiffParameter = 1./Peclet_number;
    
    for (; cell!=endc; ++cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
	local_mass_matrix = 0; local_stiffness_matrix = 0;
        fe_concentr_values.reinit (cell);
	
	double coef_a1 = 1.0;
	double coef_a2 = time_step;
	
	double old_time_step = time_step;
	
	if (use_bdf2_scheme == true)
	coef_a1 = (2*time_step + old_time_step) /
		  (time_step + old_time_step);
		  
        for (unsigned int q=0; q<n_q_points; ++q)
        {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
	    {
	      grad_phi_T[k] = fe_concentr_values.shape_grad (k,q);
	      phi_T[k] = fe_concentr_values.shape_value (k, q);
	    }
	    
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	    {
	      for (unsigned int j=0; j<dofs_per_cell; ++j)
	      {
		  local_mass_matrix(i,j) +=  coef_a1 *
					      phi_T[i] *
					      phi_T[j] *
					      fe_concentr_values.JxW(q);
					      
		  local_stiffness_matrix(i,j) +=   coef_a2*
						    DiffParameter *
						    grad_phi_T[i] *
						    grad_phi_T[j] *
						    fe_concentr_values.JxW(q);
							
	      }
	    }
	}
	
        cell->get_dof_indices (local_dofs_indices);

        constraints_concentr.distribute_local_to_global (local_stiffness_matrix,
                                                         local_dofs_indices,
                                                         matrix_concentr);
	
	constraints_concentr.distribute_local_to_global (local_mass_matrix,
                                                         local_dofs_indices,
                                                         matrix_concentr);
	
    }


}


template <int dim>
void Re_MTE_At_UBC<dim>::assemble_rhs_vector_for_mass ()
{
    const bool use_bdf2_scheme = (timestep_number != 0);
    rhs_concentr = 0;
    double nu_max = 0.0;

    const QGauss<dim> quadrature_formula (fe_concentr.get_degree() +1 );

    FEValues<dim>     fe_concentr_values (fe_concentr, quadrature_formula,
                                               update_values    |
                                               update_gradients |
                                               update_hessians  |
                                               update_quadrature_points  |
                                               update_JxW_values);

    FEValues<dim>     fe_values_vel (fe_velocity_on_concentr, quadrature_formula,
                                               update_values    |
                                               update_gradients );

    const unsigned int   dofs_per_cell   = fe_concentr.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<Tensor<1,dim> > velocity_values (n_q_points);
    std::vector<Tensor<1,dim> > old_velocity_values (n_q_points);

    std::vector<double>         concentr_values (n_q_points);
    std::vector<double>         old_concentr_values(n_q_points);
    std::vector<Tensor<1,dim> > concentr_grads(n_q_points);
    std::vector<Tensor<1,dim> > old_concentr_grads(n_q_points);
    std::vector<double>         concentr_laplacians(n_q_points);
    std::vector<double>         old_concentr_laplacians(n_q_points);

    std::vector<double> 	 gamma_values (n_q_points);

    std::vector<double>         phi_T      (dofs_per_cell);
    std::vector<Tensor<1,dim> > grad_phi_T (dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);

    const std::pair<double,double>
      global_c_range = get_extrapolated_concentr_range();

    const double average_concentr = 0.5 * (global_c_range.first +
                                           global_c_range.second);
    const double global_entropy_variation =
      get_entropy_variation_concentr (average_concentr);

    typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler_concentr.begin_active(),
	  endc = dof_handler_concentr.end(),
	  velocity_cell = dof_handler_velocity_on_concentr.begin_active();

    for (; cell!=endc; ++cell, ++velocity_cell)
    if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
    {
        local_rhs = 0;

        fe_concentr_values.reinit (cell);
        fe_values_vel.reinit (velocity_cell);

        fe_concentr_values.get_function_values (concentr_solution,
                                                concentr_values);

        fe_concentr_values.get_function_values (old_concentr_solution,
                                                old_concentr_values);

        fe_concentr_values.get_function_gradients (concentr_solution,
                                                   concentr_grads);
        fe_concentr_values.get_function_gradients (old_concentr_solution,
                                                   old_concentr_grads);

        fe_concentr_values.get_function_laplacians (concentr_solution,
                                                    concentr_laplacians);

        fe_concentr_values.get_function_laplacians (old_concentr_solution,
                                                    old_concentr_laplacians);

        fe_values_vel[velocities].get_function_values (vel_nPlus_con,
                                                       velocity_values);

        fe_values_vel[velocities].get_function_values (vel_n_con,
                                                       old_velocity_values);

	double old_time_step = time_step;

	double coef_a1 = 1.0;
	double coef_a2 = time_step;
	
	stabilization_beta = 0.05;
	stabilization_c_R = 0.5;
	stabilization_alpha = 2;
	
	double cmax = stabilization_beta;
	double cr = stabilization_c_R;
	
        double nu
	 = compute_viscosity(	concentr_values,
				old_concentr_values,
				concentr_grads,
				old_concentr_grads,
				concentr_laplacians,
				old_concentr_laplacians,
				velocity_values,
				old_velocity_values,
				gamma_values,
				maximal_velocity,
				global_c_range.second - global_c_range.first,
				0.5*(global_c_range.second + global_c_range.first),
				global_entropy_variation,
				cell->diameter(),
				cmax, cr);
// 				cell->minimum_vertex_distance());
	 
	nu_max = std::max (nu_max, nu);

	if (use_bdf2_scheme == true)
	coef_a1 = (2*time_step + old_time_step) /
		  (time_step + old_time_step);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
                grad_phi_T[k] = fe_concentr_values.shape_grad (k,q);
                phi_T[k]      = fe_concentr_values.shape_value (k, q);
            }

            const double Ts
			      =   (use_bdf2_scheme ?
				  (concentr_values[q] *
				  (time_step + old_time_step) / old_time_step
				    -
				   old_concentr_values[q] *
				   (time_step * time_step) /
				   (old_time_step * (time_step + old_time_step)))
				    :
				    concentr_values[q]);

	    const Tensor<1,dim> ext_grad_T
			      =   (use_bdf2_scheme ?
				  (concentr_grads[q] *
				  (1+time_step/old_time_step)
				    -
				   old_concentr_grads[q] *
				   time_step / old_time_step)
				    :
				    concentr_grads[q]);
			      
	    const double ext_T
			      =   (use_bdf2_scheme ?
				  (concentr_values[q] *
				  (1+time_step/old_time_step)
				    -
				   old_concentr_values[q] *
				   time_step / old_time_step)
				    :
				    concentr_values[q]);

	    const Tensor<1,dim> extrapolated_u
			      =   (use_bdf2_scheme ?
				  (velocity_values[q] * (1+time_step/old_time_step) -
 				   old_velocity_values[q] * time_step/old_time_step)
				    :
				   velocity_values[q]);

            for (unsigned int i=0; i<dofs_per_cell; ++i)
	    {

		local_rhs(i) += (Ts * phi_T[i]
				  -
				  time_step *
				  extrapolated_u * ext_grad_T * phi_T[i]
				  -
				  time_step *
				  nu * ext_grad_T * grad_phi_T[i])*
				  fe_concentr_values.JxW(q);

	    }
	}

        cell->get_dof_indices (local_dofs_indices);

        constraints_concentr.distribute_local_to_global (   local_rhs,
                                                            local_dofs_indices,
                                                            rhs_concentr);
    }  
    
}


template <int dim>
void Re_MTE_At_UBC<dim>::concentr_solve ()
{
	unsigned int n_c = dof_handler_concentr.n_dofs();
	unsigned int local_dofs =
	    DoFTools::count_dofs_with_subdomain_association (dof_handler_concentr,
	    Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
	Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);

        TrilinosWrappers::MPI::Vector distibuted_solution (map);
	
	if (num_pars > 0)
	{
	  std::map<unsigned int,double> boundary_values;
	  
	  MappingQ<dim> ff (fe_concentr.get_degree());
	  std::vector<Point<dim> > sp (dof_handler_concentr.n_dofs());
	  DoFTools::map_dofs_to_support_points (ff, dof_handler_concentr, sp);
	
	  for (unsigned int i=0; i<dof_handler_concentr.n_dofs(); ++i)
	  {
	    Point<dim> c = sp[i];
	    std::pair<unsigned int,double> distant_of_par = distant_from_particles (c);
	    if (distant_of_par.second <0) boundary_values[i] = 10.0;
	  }
	  
	  MatrixTools::apply_boundary_values (	boundary_values,
						matrix_concentr,
						distibuted_solution,
						rhs_concentr,
						false);
	    
	}
	
	matrix_concentr.compress(VectorOperation::add);
	rhs_concentr.compress(VectorOperation::add);    

	double error_crit = error_crit_MassTrn;

        SolverControl solver_control (matrix_concentr.m(),
                                      error_crit);
	SolverCG<TrilinosWrappers::MPI::Vector > cg (solver_control);

        TrilinosWrappers::PreconditionAMG preconditioner;
	preconditioner.initialize (matrix_concentr);

        cg.solve ( matrix_concentr, distibuted_solution,
                   rhs_concentr, preconditioner);

        concentr_solution = distibuted_solution;

        constraints_concentr.distribute (concentr_solution);

}

template <int dim>
std::pair<double,double>
Re_MTE_At_UBC<dim>::get_extrapolated_concentr_range ()
{
    double min_concentration, max_concentration;

    const QIterated<dim> quadrature_formula (QTrapez<1>(), (fe_pressure.get_degree()+1) + 1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (fe_concentr, quadrature_formula,
                             update_values);
    std::vector<double> concentr_values(n_q_points);
    std::vector<double> old_concentr_values(n_q_points);

    if (timestep_number != 0)
    {
        min_concentration = (1. + time_step/old_time_step) *
                        concentr_solution.linfty_norm()
                        +
                        time_step/old_time_step *
                        old_concentr_solution.linfty_norm(),
        max_concentration = -min_concentration;

        typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler_concentr.begin_active(),
            endc = dof_handler_concentr.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            fe_values.get_function_values (concentr_solution,
                                           concentr_values);
            fe_values.get_function_values (old_concentr_solution,
                                           old_concentr_values);
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                const double concentration =
                (1. + time_step/old_time_step) * concentr_values[q]-
                time_step/old_time_step * old_concentr_values[q];

                min_concentration = std::min (min_concentration, concentration);
                max_concentration = std::max (max_concentration, concentration);
            }
        }

        return std::make_pair(min_concentration, max_concentration);
    }
    else
    {
        min_concentration = concentr_solution.linfty_norm(),
        max_concentration = -min_concentration;

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_concentr.begin_active(),
        endc = dof_handler_concentr.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            fe_values.get_function_values (concentr_solution,
                                           concentr_values);

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                const double concentration = concentr_values[q];

                min_concentration = std::min (min_concentration, concentration);
                max_concentration = std::max (max_concentration, concentration);
            }
        }

        return std::make_pair(min_concentration, max_concentration);
    }
}

template <int dim>
double
Re_MTE_At_UBC<dim>::get_entropy_variation_concentr (const double average_concentr) const
  {
    if (stabilization_alpha != 2)
      return 1.;

    const QGauss<dim> quadrature_formula ( (fe_pressure.get_degree()+1) + 1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (fe_concentr, quadrature_formula,
			     update_values | update_JxW_values);
    std::vector<double> concentr_values(n_q_points);
    std::vector<double> old_concentr_values(n_q_points);

    double min_entropy = std::numeric_limits<double>::max(),
	   max_entropy = -std::numeric_limits<double>::max(),
		  area = 0,
    entropy_integrated = 0;

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_concentr.begin_active(),
      endc = dof_handler_concentr.end();
    for (; cell!=endc; ++cell)
      if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
	{
	  fe_values.reinit (cell);
	  fe_values.get_function_values (concentr_solution,
					 concentr_values);
	  fe_values.get_function_values (old_concentr_solution,
					 old_concentr_values);

	  for (unsigned int q=0; q<n_q_points; ++q)
	    {
	      const double T = (concentr_values[q] +
				old_concentr_values[q]) / 2;
	      const double entropy = ((T-average_concentr) *
				      (T-average_concentr));

	      min_entropy = std::min (min_entropy, entropy);
	      max_entropy = std::max (max_entropy, entropy);
	      area += fe_values.JxW(q);
	      entropy_integrated += fe_values.JxW(q) * entropy;
	    }
	}

    const double local_sums[2]   = { entropy_integrated, area },
		 local_maxima[2] = { -min_entropy, max_entropy };
    double global_sums[2], global_maxima[2];

    Utilities::MPI::sum (local_sums,   MPI_COMM_WORLD, global_sums);
    Utilities::MPI::max (local_maxima, MPI_COMM_WORLD, global_maxima);


    const double average_entropy = global_sums[0] / global_sums[1];
    const double entropy_diff = std::max(global_maxima[1] - average_entropy,
					 average_entropy - (-global_maxima[0]));
    return entropy_diff;
}

template <int dim>
double
Re_MTE_At_UBC<dim>::
  compute_viscosity(	const std::vector<double>		&concentr,
                       const std::vector<double>          	&old_concentr,
                       const std::vector<Tensor<1,dim> >  	&concentr_grads,
                       const std::vector<Tensor<1,dim> >  	&old_concentr_grads,
                       const std::vector<double>          	&concentr_laplacians,
                       const std::vector<double>          	&old_concentr_laplacians,
                       const std::vector<Tensor<1,dim> >  	&velocity_values,
                       const std::vector<Tensor<1,dim> >  	&old_velocity_values,
                       const std::vector<double>          	&gamma_values,
                       const double                       	global_u_infty,
                       const double                       	global_i_variation,
			const double				average_levelset,
			const double 				global_entropy_variation,
                       const double                       	cell_diameter,
			double 					stabilization_beta,
			double					stabilization_c_R)
{

    if (global_u_infty == 0)
      return 5e-3 * cell_diameter;

    const unsigned int n_q_points = concentr.size();

    double max_residual = 0;
    double max_velocity = 0;

    for (unsigned int q=0; q < n_q_points; ++q)
      {
	const Tensor<1,dim> u = (velocity_values[q] +
				old_velocity_values[q]) / 2;
	const double T = (concentr[q] + old_concentr[q]) / 2;
	const double dT_dt = (concentr[q] - old_concentr[q])
			     / old_time_step;
	const double u_grad_T = u * (concentr_grads[q] +
				     old_concentr_grads[q]) / 2;


	double residual
	  = std::abs(dT_dt + u_grad_T);

	if (stabilization_alpha == 2)
	  residual *= std::abs(T - average_levelset);

	max_residual = std::max (residual,        max_residual);
	max_velocity = std::max (std::sqrt (u*u), max_velocity);
      }

    const double max_viscosity = (stabilization_beta *
				  max_velocity * cell_diameter);
    if (timestep_number == 0)
      return max_viscosity;
    else
      {
	Assert (old_time_step > 0, ExcInternalError());

	double entropy_viscosity;
	if (stabilization_alpha == 2)
	  entropy_viscosity = (stabilization_c_R *
			       cell_diameter * cell_diameter *
			       max_residual /
			       global_entropy_variation);
	else
	  entropy_viscosity = (stabilization_c_R *
			       cell_diameter * global_Omega_diameter *
			       max_velocity * max_residual /
			       (global_u_infty * global_i_variation));

	return std::min (max_viscosity, entropy_viscosity);
      }
}

template <int dim>
void Re_MTE_At_UBC<dim>::compute_vorticity ()
{
    pcout << "* Compute Vorticity.. ";

    matrix_pressure = 0;
    rhs_pressure = 0;

    const QGauss<dim> quadrature_formula(fe_pressure.get_degree()+1 + 1);

    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);

    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
                                      update_values |
                                      update_gradients);

    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);

    Vector<double> local_rhs (dofs_per_cell);

    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

    std::vector<std::vector<Tensor<1,dim> > >
	grad_vel_n_plus_1_values (n_q_points, std::vector<Tensor<1,dim> >(dim));

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_pressure.begin_active(),
    endc = dof_handler_pressure.end();

    typename DoFHandler<dim>::active_cell_iterator
    vel_cell = dof_handler_velocity.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell)
        if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
        {
            fe_values_pressure.reinit (cell);
            fe_values_velocity.reinit (vel_cell);

            fe_values_velocity.get_function_gradients(vel_n, grad_vel_n_plus_1_values);

            cell->get_dof_indices (local_dofs_indices);

            local_matrix = 0;
            local_rhs = 0;

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
		    local_matrix(i,j)   +=  fe_values_pressure.shape_value(i, q)*
                                            fe_values_pressure.shape_value(j, q)*
                                            fe_values_pressure.JxW(q);

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    local_rhs(i) += fe_values_pressure.shape_value(i, q)*
				      std::abs
                                    (	grad_vel_n_plus_1_values[q][0][1] -
					grad_vel_n_plus_1_values[q][1][0])*
                                    fe_values_pressure.JxW(q);

            }

            constraint_pressure.distribute_local_to_global (local_matrix,
                                                            local_dofs_indices,
                                                            matrix_pressure);

            constraint_pressure.distribute_local_to_global (local_rhs,
                                                            local_dofs_indices,
                                                            rhs_pressure);
        }


    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_pressure,
			      Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::MPI::Vector distibuted_xx (map);

    matrix_pressure.compress(VectorOperation::add);
    rhs_pressure.compress(VectorOperation::add);

    SolverControl solver_control (matrix_pressure.m(), error_crit_NS*rhs_pressure.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionILU preconditioner;
    preconditioner.initialize (matrix_pressure);
    cg.solve (matrix_pressure, distibuted_xx, rhs_pressure, preconditioner);
    vorticity = distibuted_xx;

    pcout   << solver_control.last_step()
            << std::endl;

    constraint_pressure.distribute (vorticity);  
}

template <int dim>
void Re_MTE_At_UBC<dim>::pars_move (std::ofstream &out_ppp, std::ofstream &out_vel)
{
	pcout << "* Large Particle Advancement.. " << std::endl;
	QGauss<dim>  quadrature_formula(4);
	const unsigned int n_q_points = quadrature_formula.size();

	FEValues<dim> fe_velocity_values (fe_velocity, quadrature_formula,
					UpdateFlags(update_values    |
					update_gradients |
					update_q_points  |
					update_JxW_values));

	const unsigned int   dofs_per_cell = fe_velocity.dofs_per_cell;
	std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator  cell, endc;

	cell = dof_handler_velocity.begin_active();
	endc = dof_handler_velocity.end();

	std::vector<Vector<double> > vel_solu (n_q_points , Vector<double>(dim));
	std::vector<std::vector<Tensor<1,dim> > > vel_solGrads (fe_velocity_values.n_quadrature_points,
                                                  std::vector<Tensor<1,dim> > (dim));
	std::vector<Point<dim> > totVel;
	double totvis[1000];

	for (unsigned int n=0; n<num_pars; ++n)
	{
	    Point<dim> aa;
	    totVel.push_back (aa);
	    totvis[n] = 0.0;
	}

	for (; cell!=endc; ++cell)
	{
		fe_velocity_values.reinit (cell);
		fe_velocity_values.get_function_values (vel_n_plus_1, vel_solu);
		fe_velocity_values.get_function_gradients (vel_n_plus_1 , vel_solGrads);

		cell->get_dof_indices (local_dofs_indices);

		Point<dim> c = cell->center();
		std::pair<unsigned int,double> distant_of_par = distant_from_particles (c);

		if (distant_of_par.second <0)
		for (unsigned int q=0 ; q < n_q_points ; ++q)
		for (unsigned int i=0 ; i < n_q_points ; ++i)
		{
                    unsigned int nn_par = distant_of_par.first;

		    totvis[nn_par] +=	fe_velocity_values.shape_value (i,q) *
					fe_velocity_values.JxW(q);

                    for (unsigned d=0; d<dim; ++d)
			totVel[nn_par][d] +=	fe_velocity_values.shape_value (i,q) *
						fe_velocity_values.JxW(q)*
						vel_solu[q](d);
		}
	}

	double half_x = 0.5*(xmax+xmin);
	double x_leng = std::abs(xmax-xmin);

	out_ppp << timestep_number << " ";
	out_vel << timestep_number << " ";
	
	for (unsigned n=0; n<num_pars; ++n)
	{
		cenPar[n][0] += (totVel[n][0] / totvis[n]) * time_step;
		cenPar[n][1] += (totVel[n][1] / totvis[n]) * time_step;

		if (cenPar[n][0] < xmin)  cenPar[n][0] = cenPar [n][0] + x_leng;
		if (cenPar[n][0] >= xmax)  cenPar[n][0] = cenPar [n][0] - x_leng;

		image_cenPar [n] = cenPar [n];

		if (cenPar[n][0] < half_x)  image_cenPar [n][0] = cenPar [n][0] + x_leng;
		if (cenPar[n][0] >= half_x)  image_cenPar [n][0] = cenPar [n][0] - x_leng;

		Point<dim> tmpPar;

		double xlen = std::abs(xmax-xmin);

// 		pcout	<< n << " "
// 			<< totVel[n]/ totvis[n]  << " "
// 			<< cenPar[n] << " "
// 			<< image_cenPar[n]
// 			<< std::endl;

		out_ppp << cenPar[n] << " ";
		out_vel << (totVel[n][0] / totvis[n]) << " " << (totVel[n][1] / totvis[n]) << " ";
	}
	out_ppp << std::endl;
	out_vel << std::endl;

}

template <int dim>
void Re_MTE_At_UBC<dim>::pars_angular (std::ofstream &out_ang)
{
	QGauss<dim>  quadrature_formula(4);
	const unsigned int n_q_points = quadrature_formula.size();

	FEValues<dim> fe_velocity_values (fe_velocity, quadrature_formula,
					UpdateFlags(update_values    |
					update_gradients |
					update_q_points  |
					update_JxW_values));

	const unsigned int   dofs_per_cell = fe_velocity.dofs_per_cell;
	std::vector<unsigned int> local_dofs_indices (dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator  cell, endc;

	cell = dof_handler_velocity.begin_active();
	endc = dof_handler_velocity.end();

	std::vector<Vector<double> > vel_solu (n_q_points , Vector<double>(dim));
	std::vector<std::vector<Tensor<1,dim> > > vel_solGrads (fe_velocity_values.n_quadrature_points,
                                                  std::vector<Tensor<1,dim> > (dim));
	std::vector<Point<dim> > totVel;
	double totvis[1000];

	for (unsigned int n=0; n<num_pars; ++n)
	{
	    Point<dim> aa;
	    totVel.push_back (aa);
	    totvis[n] = 0.0;
	}

	double Bot = 0;
	double Upper = 0;

	for (; cell!=endc; ++cell)
	{
		fe_velocity_values.reinit (cell);
		fe_velocity_values.get_function_values (vel_n_plus_1, vel_solu);
		fe_velocity_values.get_function_gradients (vel_n_plus_1 , vel_solGrads);

		cell->get_dof_indices (local_dofs_indices);
		std::vector<Point<dim> > coor = fe_velocity_values.get_quadrature_points();

		for (unsigned int q=0; q<n_q_points; ++q)
		  for (unsigned int d=0; d<dim; ++d)
		    coor[q][d] = coor[q][d] - cenPar[0][d];

		Point<dim> c = cell->center();
		std::pair<unsigned int,double> distant_of_par = distant_from_particles (c);

		if (distant_of_par.second <0)
		for (unsigned int q=0 ; q < n_q_points ; ++q)
		for (unsigned int i=0 ; i < n_q_points ; ++i)
		{
                      unsigned int in_i = fe_velocity.component_to_system_index (0,i);

		      Upper += fe_velocity_values.shape_value (in_i, q) *
				(
				+       vel_solu[q][1] * coor[q][0]
				-       vel_solu[q][0] * coor[q][1]
				)
				* fe_velocity_values.JxW(q);

		      Bot  += fe_velocity_values.shape_value (in_i,q) *
			      (
			      +       coor[q][0] * coor[q][0]
			      +       coor[q][1] * coor[q][1]
			      )
			      * fe_velocity_values.JxW(q);
		}
	}

	out_ang	<< timestep_number << " "
		<< timestep_number*time_step << " "
		<< Upper/Bot << std::endl;
}


template <int dim>
double Re_MTE_At_UBC<dim>::get_maximal_velocity () const
{
     const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                              fe_pressure.get_degree () +1 +1);
     const unsigned int n_q_points = quadrature_formula.size();

     FEValues<dim> fe_values (fe_velocity, quadrature_formula, update_values);
     std::vector<Tensor<1,dim> > velocity_values(n_q_points);
     double max_velocity = 0;

     const FEValuesExtractors::Vector velocities (0);

     typename DoFHandler<dim>::active_cell_iterator
       cell = dof_handler_velocity.begin_active(),
       endc = dof_handler_velocity.end();
     for (; cell!=endc; ++cell)
       {
         fe_values.reinit (cell);
         fe_values[velocities].get_function_values (vel_n_plus_1,
                                                    velocity_values);

         for (unsigned int q=0; q<n_q_points; ++q)
           max_velocity = std::max (max_velocity, velocity_values[q].norm());
       }

     return max_velocity;
}
 
template <int dim>
void Re_MTE_At_UBC<dim>::particle_generation ()
{
    pcout << "* Particle Generation.. " << std::endl;

    double xlen = std::abs(xmax-xmin);
    double ylen = std::abs(ymax-ymin);
    double zlen = std::abs(zmax-zmin);
    
    if (is_random_particles == false)
    {
        std::string filename_par = "particle.prm";
        std::ifstream out_par (filename_par.c_str());

	pcout << "* Load Particle Information..." << std::endl;
        for (unsigned int n = 0; n < num_pars; ++n)
        {
            double co1, co2, co3, ra;

            Point<dim> raco, image_raco;
            if (dim == 2) out_par >> co1 >> co2;
            if (dim == 3) out_par >> co1 >> co2 >> co3;

            raco[0] = co1;
            raco[1] = co2;
            if (dim == 3) raco[2] = co3;

            image_raco = raco;

            if (raco[0] < 0.5*(xmax+xmin) ) image_raco[0] = raco[0] + xlen;
            if (raco[0] > 0.5*(xmax+xmin) ) image_raco[0] = raco[0] - xlen;

            cenPar.push_back(raco);
            image_cenPar.push_back(image_raco);

            pcout <<  cenPar[n] << " | " << image_cenPar[n] << " | " << par_rad << std::endl;
        }
    }


    if (is_random_particles == true)
    {
        pcout << "* Generate Random Particles.. " << std::endl;

	std::string filename_par = "random_generated_particle.dat";
        std::ofstream out_ini_par (filename_par.c_str());

        std::vector<unsigned int> dum;

        dum.push_back (int(xlen)*10000);
        dum.push_back (int(ylen)*10000);
        if (dim == 3) dum.push_back (int(zlen)*10000);

        for (unsigned int n = 0 ; n < num_pars ; ++n)
        {
            Point<dim> tmpPar;
            Point<dim> tmpp;
            cenPar.push_back (tmpPar);
            image_cenPar.push_back (tmpPar);

            bool valid_tmp_pars;
            valid_tmp_pars = false;

            unsigned int count = 0;
            do
            {
                bool valid_gen_tmpPar;

                unsigned int t = 0;
                do
                {
                    valid_gen_tmpPar = true;
                    for (unsigned int d = 0; d < dim ; ++d)
                    {
                        unsigned int num0 = rand()%dum[d];
                        double num1 = num0;
                        double leg_max = 0.0;
                        if (d == 0) leg_max = xmax;
                        if (d == 1) leg_max = ymax;
                        if (d == 2) leg_max = zmax;
                        num1 = num1/10000 - leg_max;
//                         num1 = num1/10000;
                        tmpPar[d] = num1;
                    }

		    double xxmin = xmin + par_rad;
		    double xxmax = xmax - par_rad;
		    double yymin = ymin + par_rad + thr_val_particle_dist;
		    double yymax = ymax - par_rad - thr_val_particle_dist;
		    
		    double y_center1 = 0.0 + (par_rad + 0.005);
		    double y_center2 = 0.0 - (par_rad + 0.005);
		    
		    double half_y_len = 0.5*(ymax+ymin);
		    double off_yymin = half_y_len + par_rad;
		    double off_yymax = half_y_len - par_rad;

		    bool crit1, crit2, crit3;
		    crit1 = (tmpPar[0]  < xxmin) || (tmpPar[0] > xxmax);
		    crit2 = (tmpPar[1]  < yymin) || (tmpPar[1] > yymax);
		    crit3 = (tmpPar[1]  < y_center1) && (tmpPar[1] > y_center2);
		    
                   if (crit1 || crit2 || crit3) valid_gen_tmpPar = false;

                    ++t;
                } while (valid_gen_tmpPar == false);


		  cenPar[n] = tmpPar;
		  image_cenPar[n] = tmpPar;
		  double half_xlen = 0.5*(xmax+xmin);

		  if (cenPar[n][0] < half_xlen)  image_cenPar[n][0] = tmpPar[0] + xlen;
		  if (cenPar[n][0] > half_xlen)  image_cenPar[n][0] = tmpPar[0] - xlen;

		  valid_tmp_pars = true;

		  for(unsigned i = 0 ; i < n ; ++i)
		  {
                    double dist_cen_cen = 0.0;

                    dist_cen_cen = cenPar[i].distance(tmpPar);

                    if (dist_cen_cen < 2*par_rad) valid_tmp_pars = false;

                    dist_cen_cen = image_cenPar[i].distance(tmpPar);

                    if (dist_cen_cen < 2*par_rad) valid_tmp_pars = false;
		  }

		  if (n == 0) valid_tmp_pars = true;

		  ++count;

            } while (valid_tmp_pars == false); //do
            pcout <<  cenPar[n] << std::endl;
	    out_ini_par <<  cenPar[n] << std::endl;
        }
    }
}

template <int dim>
void Re_MTE_At_UBC<dim>::plotting_solution_flow (unsigned int np)
{
  if (is_verbal_output == true)
      pcout << "* Plot the solutions...for Flow"  << std::endl;

//     compute_vorticity (); 
//     find_the_streamlines ();
    
    const FESystem<dim> joint_fe (fe_velocity, 1,
				  fe_pressure, 1,
				  fe_pressure, 1,
				  fe_pressure, 1
 				);

    DoFHandler<dim> joint_dof_handler (triangulation);
    joint_dof_handler.distribute_dofs (joint_fe);
    Assert (joint_dof_handler.n_dofs() ==
	    dof_handler_velocity.n_dofs() +
	    3*dof_handler_pressure.n_dofs(),
	    ExcInternalError());

    Vector<double> joint_solution (joint_dof_handler.n_dofs());

    {
        std::vector<unsigned int> local_joint_dof_indices (joint_fe.dofs_per_cell);
        std::vector<unsigned int> local_vel_dof_indices (fe_velocity.dofs_per_cell);
	std::vector<unsigned int> local_pre_dof_indices (fe_pressure.dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator
        joint_cell  = joint_dof_handler.begin_active(),
        joint_endc  = joint_dof_handler.end(),
        vel_cell    = dof_handler_velocity.begin_active(),
        pre_cell    = dof_handler_pressure.begin_active();

        for (; joint_cell!=joint_endc; ++joint_cell, ++vel_cell, ++pre_cell)
        {
	    joint_cell -> get_dof_indices (local_joint_dof_indices);
	    vel_cell-> get_dof_indices (local_vel_dof_indices);
	    pre_cell->get_dof_indices (local_pre_dof_indices);

            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
	    {
                if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                    joint_solution(local_joint_dof_indices[i])
                    = vel_n_plus_1 
			(local_vel_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else if (joint_fe.system_to_base_index(i).first.first == 1)
                {
                    joint_solution(local_joint_dof_indices[i])
                    = particle_distb 
			(local_pre_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else if (joint_fe.system_to_base_index(i).first.first == 2)
                {
                    joint_solution(local_joint_dof_indices[i])
                    = pre_n_plus_1 
			(local_pre_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else if (joint_fe.system_to_base_index(i).first.first == 3)
                {
                    joint_solution(local_joint_dof_indices[i])
                    = vorticity 
			(local_pre_dof_indices[joint_fe.system_to_base_index(i).second]);
		}
//                 else if (joint_fe.system_to_base_index(i).first.first == 4)
//                 {
//                     joint_solution(local_joint_dof_indices[i])
//                     = stream_func 
// 			(local_pre_dof_indices[joint_fe.system_to_base_index(i).second]);
//                 }
	    }
        }
    }

    std::vector<std::string> joint_solution_names (dim, "v");
    joint_solution_names.push_back ("particle_distb");  
    joint_solution_names.push_back ("pressure");
    joint_solution_names.push_back ("vorticity");
//     joint_solution_names.push_back ("stream_func");

    
    DataOut<dim> data_out;

    data_out.attach_dof_handler (joint_dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
      (dim + 3, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i=0; i<dim; ++i)
        data_component_interpretation[i]
        = DataComponentInterpretation::component_is_part_of_vector;

    data_out.add_data_vector (joint_solution, joint_solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches (fe_pressure.get_degree()+1);

    std::ostringstream filename;
    filename << "s-flow-" << Utilities::int_to_string(np, 4) << ".vtu";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);
}


template <int dim>
void Re_MTE_At_UBC<dim>::plotting_solution_mass (unsigned int np)
{
  if (is_verbal_output == true)
      pcout << "* Plot the solutions...for Mass"  << std::endl;
    
    const FESystem<dim> joint_fe ( fe_velocity_on_concentr, 1,
				    fe_concentr, 1);

    DoFHandler<dim> joint_dof_handler (triangulation_concentr);
    joint_dof_handler.distribute_dofs (joint_fe);
    Assert (joint_dof_handler.n_dofs() ==
	    dof_handler_velocity_on_concentr() +
	    dof_handler_concentr.n_dofs(),
	    ExcInternalError());

    Vector<double> joint_solution (joint_dof_handler.n_dofs());

    {
        std::vector<unsigned int> local_joint_dof_indices (joint_fe.dofs_per_cell);
        std::vector<unsigned int> local_vel_dof_indices (fe_velocity_on_concentr.dofs_per_cell);
	std::vector<unsigned int> local_con_dof_indices (fe_concentr.dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator
	  joint_cell  = joint_dof_handler.begin_active(),
	  joint_endc  = joint_dof_handler.end(),
	  vel_cell    = dof_handler_velocity_on_concentr.begin_active(),
	  con_cell    = dof_handler_concentr.begin_active();

        for (; joint_cell!=joint_endc; ++joint_cell, ++vel_cell, ++con_cell)
        {
	    joint_cell -> get_dof_indices (local_joint_dof_indices);
	    vel_cell-> get_dof_indices (local_vel_dof_indices);
	    con_cell->get_dof_indices (local_con_dof_indices);

            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
	    {
                if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                    joint_solution(local_joint_dof_indices[i])
                    = vel_nPlus_con 
			(local_vel_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else if (joint_fe.system_to_base_index(i).first.first == 1)
                {
                    joint_solution(local_joint_dof_indices[i])
                    = concentr_solution 
			(local_con_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
	    }
        }
    }

    std::vector<std::string> joint_solution_names (dim, "v");  
    joint_solution_names.push_back ("c");

    DataOut<dim> data_out;

    data_out.attach_dof_handler (joint_dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
      (dim + 1, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i=0; i<dim; ++i)
        data_component_interpretation[i]
        = DataComponentInterpretation::component_is_part_of_vector;

    data_out.add_data_vector (joint_solution, joint_solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches (fe_concentr.get_degree());

    std::ostringstream filename;
    filename << "s-mass-" << Utilities::int_to_string(np, 4) << ".vtu";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);
}

template <int dim>
void Re_MTE_At_UBC<dim>::compute_quantities (std::ofstream &out_diff, double total_time)
{
//     for (unsigned int i=0; i<num_of_tracers; ++i)
//     output << tracer_position[i] << " 0" << std::endl;  

  double max_distance_y = 0.0;
  double avr_distance_y = 0.0;
  Point<dim> avr_velocity;
  
  if (is_passive_tracer == true)
  for (unsigned int i=0; i<num_of_tracers; ++i)
  {
    double local_dist_y = std::abs(tracer_position[i][1]);
    avr_distance_y += std::abs(tracer_position[i][1])/double(num_of_tracers);
    
    avr_velocity[1] += std::abs(fluid_velocity_at_tracer[i][1])/double(num_of_tracers);
    avr_velocity[0] += std::abs(fluid_velocity_at_tracer[i][0] - tracer_position[i][1])/double(num_of_tracers);
    
    max_distance_y = std::max(local_dist_y, max_distance_y);
  }

  double avr_effective_viscosity = 0.0;

  {      
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             fe_pressure.get_degree() +1 +1);
    
    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);
    
    const unsigned int dofs_per_cell = fe_velocity.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<std::vector<Tensor<1,dim> > > grad_vel_values 
						(fe_values_velocity.n_quadrature_points ,
                                              std::vector<Tensor<1,dim> > (dim));
    
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();
          
    for (; cell!=endc; ++cell)
    {
      double nu = 1.0;
      
      {
	Point<dim> c = cell->center();
	std::pair<unsigned int,double> distant_of_par = distant_from_particles (c);
	if (distant_of_par.second <0) nu = FacPar;
      }
      
      fe_values_velocity.reinit (cell);
      fe_values_velocity.get_function_gradients (vel_n_plus_1, grad_vel_values);

      double local_interg = 0.0;
      double local_area = 0.0;
      
      for (unsigned int q=0; q<n_q_points; ++q)
      {  
	Point<dim> dd;
	dd [0] = grad_vel_values[q][1][0];
	dd [1] = grad_vel_values[q][0][1];
	
	local_interg += nu*(dd[0] + dd[1])*fe_values_velocity.JxW(q);
      }
      
      avr_effective_viscosity += local_interg;
    }
  }
							      						      
  
  out_diff 	<< timestep_number << " "
		<< total_time << " "
		<< max_distance_y << " "
		<< avr_distance_y << " "
		<< avr_velocity << " "
		<< avr_effective_viscosity 
		<< std::endl;
}

template <int dim>
void Re_MTE_At_UBC<dim>::plotting_tracers (unsigned int np)
{
  pcout << "* Plotting for tracers.." << std::endl;

  std::ostringstream filename;
  filename << "s-tr-" << Utilities::int_to_string(np, 4) << ".vtu";
  std::ofstream output (filename.str().c_str());

  output << "ASCII" << std::endl;
  output << "DATASET UNSTRUCTURED_GRID " << std::endl;
  output << std::endl;
  output << "POINTS " << tracer_position.size() << " double" << std::endl;

  for (unsigned int i=0; i<num_of_tracers; ++i)
    output << tracer_position[i] << " 0" << std::endl;    
}


template <int dim>
void Re_MTE_At_UBC<dim>::find_the_streamlines ()
{
    if (is_verbal_output == true)
    pcout << "* Find Streamlines.. ";
    
    matrix_pressure = 0;
    rhs_pressure = 0;
    
    const QGauss<dim> quadrature_formula (fe_pressure.get_degree()+1);

    FEValues<dim> fe_values_pressure (fe_pressure, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);

    FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
                                      update_values |
                                      update_gradients);

    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);

    Vector<double> local_rhs (dofs_per_cell);
    std::vector<unsigned int> local_dofs_indices (dofs_per_cell);
    std::vector<std::vector<Tensor<1,dim> > >
	grad_vel_n_plus_1_values (n_q_points, std::vector<Tensor<1,dim> >(dim));

    typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler_pressure.begin_active(),
	endc = dof_handler_pressure.end(),
	vel_cell = dof_handler_velocity.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell)
        if (cell->subdomain_id() == 
	  Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
        {
            fe_values_pressure.reinit (cell);
            fe_values_velocity.reinit (vel_cell);

            fe_values_velocity.get_function_gradients(vel_n_plus_1, grad_vel_n_plus_1_values);

            cell->get_dof_indices (local_dofs_indices);

            local_matrix = 0;
            local_rhs = 0;
		      
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
		    local_matrix(i,j) += fe_values_pressure.shape_grad(i, q)*
					  fe_values_pressure.shape_grad(j, q)*
					  fe_values_pressure.JxW(q);

                for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {  
		      local_rhs(i) += 	fe_values_pressure.shape_value(i, q)*
					( grad_vel_n_plus_1_values[q][0][1] -
					  grad_vel_n_plus_1_values[q][1][0])*
					fe_values_pressure.JxW(q);
				    
		  }
            }

            constraint_pressure.distribute_local_to_global (local_matrix,
                                                            local_dofs_indices,
                                                            matrix_pressure);

            constraint_pressure.distribute_local_to_global (local_rhs,
                                                            local_dofs_indices,
                                                            rhs_pressure);
        }

    unsigned int n_p = dof_handler_pressure.n_dofs();
    unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association 
				(dof_handler_pressure,
				Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));

    Epetra_Map map (-1, local_dofs, 0 , trilinos_communicator);
    TrilinosWrappers::MPI::Vector distributed_sol (map);
    
    matrix_pressure.compress(VectorOperation::add);
    rhs_pressure.compress(VectorOperation::add);

    SolverControl solver_control (matrix_pressure.m(), error_crit_NS*rhs_pressure.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize (matrix_pressure);
    cg.solve (matrix_pressure, distributed_sol, rhs_pressure, preconditioner);
    
    if (is_verbal_output == true)
      pcout	<< solver_control.last_step()
		<< std::endl;
		
    constraint_pressure.distribute (distributed_sol);  
    stream_function = distributed_sol;
}

template <int dim>
void Re_MTE_At_UBC<dim>::get_initial_flow_state ()
{

  
  if (is_convective_flow == true)
    VectorTools::interpolate (	dof_handler_velocity,
				Inflow_Velocity<dim>(mean_velocity, Inlet_velocity_type),
				vel_n_plus_1);
  
  vel_n = vel_n_minus_1 = vel_n_plus_1;
    
  if (Inlet_velocity_type == 1 && is_convective_flow == true)
  {    
    double initial_rey_num = Reynolds_number;
    double initial_time_interval = time_step;
    
    time_step = 5e-3;
    Reynolds_number = 1.0;
  
    for (unsigned int i=0; i<particle_distb.size(); ++i) 
      particle_distb(i) = 1.0;
  
    do
    {
      extrapolation_step ();
      diffusion_step ();
      projection_step ();
//     	pressure_correction_step_rot ();
      pressure_correction_step_stn ();
      vel_pre_convergence ();
      solution_update (); 
      pcout << std::endl;
    }  while (pressure_l2_norm > 1e-5);
  
//   	plotting_solution (0);
    Reynolds_number = initial_rey_num;
    time_step = initial_time_interval;
  }
  
  turn_on_periodic_bnd = true;

}

template <int dim>
void Re_MTE_At_UBC<dim>::run ()
{
    std::ostringstream filename_ang;
    filename_ang << "particle_angular.dat";
    std::ofstream out_ang (filename_ang.str().c_str());

    std::ostringstream filename_ppp;
    filename_ppp << "particle_position.dat";
    std::ofstream out_ppp (filename_ppp.str().c_str());

    std::ostringstream filename_vel;
    filename_vel << "particle_velocity.dat";
    std::ofstream out_vel (filename_vel.str().c_str());
    
    std::ostringstream filename_trCV;
    filename_trCV << "tracer_conversion.dat";
    std::ofstream out_trCV (filename_trCV.str().c_str());
    
    std::ostringstream filename_diff;
    filename_diff << "diffusivity.dat";
    std::ofstream out_diff (filename_diff.str().c_str());
    
    srand( (unsigned) time(NULL));
    readat ();
    particle_generation ();
    create_triangulation ();
  
    setup_dofs (true, true, true);
    setup_dofs_con (true, false);
    get_initial_flow_state ();
  
    for (unsigned int i=0; i<reLev; ++i)
      refine_mesh (false);
    
    setup_dofs (true, false, true);
    setup_dofs_con (true, true);
    
    double starting_time_for_massTrn = 0.0;
    double total_time = 0.0;
    unsigned int counter = 0;
    
    double given_time_step = time_step;
    double maximal_velocity11 = 0.0;
    double given_maximal_velocity = 0.0;
    
    if (is_mass_trn == true)
    VectorTools::interpolate (	dof_handler_concentr,
				Concentration<dim>(),
				concentr_solution);
    old_concentr_solution = concentr_solution;
  
    if (is_passive_tracer == true)
    {
      assigning_tracer_distribution ();
      plotting_tracers (0);
    }    
     
    particle_distribution ();
        
    if (is_convective_flow == true) plotting_solution_flow (0);
    if (is_mass_trn == true) plotting_solution_mass (0);
        
    unsigned int pre_kk, kk;
    pre_kk = 0;
    kk = 0;
        
//    return;
    
    do
    {
	  pcout << std::endl;
	  
	  total_time += time_step;	  
	  
// 	  if (timestep_number == 0) time_step = given_time_step;
// 	  if (timestep_number > 0) 
// 	  {
// 	    maximal_velocity11 = maximal_velocity/given_maximal_velocity;
	    
// 	    time_step = given_time_step;
	    
// 	    if (maximal_velocity11 > mean_velocity)
// 	      time_step = given_time_step/maximal_velocity11;
// 	  }
	  
	  time_step = given_time_step;
	  
	  pcout <<"# Time Step = " << timestep_number << ", "
		<< total_time << ", " << final_time << " | " 
		<< maximal_velocity << ", " << maximal_velocity11 
		<< ", " << time_step <<std::endl;
		
	  if (std::abs(mean_velocity) > 1e-6 && is_convective_flow == true)
	  {
	    pcout <<"* Solve NS Eqn.." << std::endl; 
	    
	    old_time_step = time_step;
	  
	    particle_distribution ();
	    extrapolation_step ();
	    diffusion_step ();
	  
	    maximal_velocity = get_maximal_velocity ();
	    if (timestep_number == 0) 
	      given_maximal_velocity = maximal_velocity;
	  
	    projection_step ();
// 	  	pressure_correction_step_rot ();
	    pressure_correction_step_stn ();
	    vel_pre_convergence ();
	    solution_update ();
	  
	    if (is_solid_particle == true)
	    {
	      pars_angular (out_ang);
	      pars_move (out_ppp, out_vel);
	    }
	  }//if-mean_velocity>0, not pure diffusion
	  
	  if (is_passive_tracer == true)
	  {
	    double tmp_time_step = time_step;
	    time_step = 0.1*time_step;
	    
	    for (unsigned int k=0; k<10; ++k)
	    {
	      find_fluid_veloicity_at_tracer_position ();
	      advancement_for_tracer ();
	    }
	    
	    time_step = tmp_time_step;
	    
	    pre_kk = kk;
	    kk = 0;
	    for (unsigned int i=0; i<tracer_position.size(); ++i)
	    {  
	      if (what_type_tracer[i] == 3)
		kk = kk + 1;
	    }
	    pcout	<< "No. Touch At Wall = "
			<< timestep_number << " | "
			<< touch_at_ymax << " | "
			<< touch_at_ymin << " | "
			<< kk << " " 
			<< pre_kk << " " 
			<< std::abs(kk-pre_kk) << std::endl;
			
	    out_trCV	<< timestep_number << " "
			<< touch_at_ymax << " "
			<< touch_at_ymin << " "
			<< kk << " " 
			<< pre_kk << " " 
			<< std::abs(kk-pre_kk) << std::endl;
			
	  }
	  
	  if (is_mass_trn == true)
	  {
	     transfer_velocities_on_concentr ();
	     
	     pcout	<< "-- " 
			<< "Maximal Velocity = " 
			<< maximal_velocity << ", " 
			<< old_time_step << " | "
			<< time_step
			<< std::endl;
		      
	    double mm = 1*(maximal_velocity);
	    unsigned int rrr = static_cast<unsigned int> (mm) + 1;
	    if (timestep_number > 0)
		time_step = (1./static_cast<double>(rrr))*given_time_step;
	    if (timestep_number == 0) time_step = given_time_step;
	    old_time_step = time_step;
	    
	    pcout <<"* Solve Mass Eqn.." << std::endl; 
	    
	    
	    for (unsigned int i=0; i<rrr; ++i)
	    {	    
	      tmp_concentr_solution = concentr_solution;
	      assemble_matrix_for_mass ();
	      assemble_rhs_vector_for_mass ();
	      concentr_solve ();
	      old_concentr_solution = tmp_concentr_solution; ;
	    }
	    
	    time_step = given_time_step;
	    old_time_step = time_step;
	  }
	  
	  compute_quantities (out_diff, total_time);
	  
	  if (timestep_number % output_fac == 0) 
	  {
	    if (is_convective_flow == true || num_pars>0) plotting_solution_flow (counter+1);
	    if (is_mass_trn == true) plotting_solution_mass (counter+1);
	    if (is_passive_tracer == true) plotting_tracers (counter+1);
	    ++counter;
	  }
	  
	  if (timestep_number % refine_fac == 0 && is_solid_particle == true && reLev > 0) 
	    refine_mesh (true);
// 	  break;
	  ++timestep_number;
	  
	  pcout << std::endl;

    }  while (total_time < final_time);
}

int main (int argc, char *argv[])
{
    try
    {
	deallog.depth_console (0);
	Utilities::System::MPI_InitFinalize mpi_initialization(argc, argv);

	ParameterHandler  prm;
        ParameterReader   param(prm);
        param.read_parameters("input.prm");

        prm.enter_subsection ("Problem Definition");
	  unsigned int dimn = prm.get_integer ("Dimension");
        prm.leave_subsection ();
	
        if (dimn == 2)
        {
            Re_MTE_At_UBC<2>  ysh (prm);
            ysh.run ();
        } else if (dimn == 3)
        {
            Re_MTE_At_UBC<3>  ysh (prm);
            ysh.run ();
        };
    }

    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }

    return 0;
}
