/*
   This file takes input .tgf, .obj and outputs tetrahedralized mesh .mesh, Euclidean weights .dmat
 
 
   It also visualizes the bones and weight colors.
   Press 1-2-3-4-5-6-7-8-9 to visualize volumetric mesh
   Press E-R to change the selected bone to visualize its associated colors
 
 */

#include <assert.h>
#include <string>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>

#include <igl/writeMESH.h>
#include <igl/writeDMAT.h>
#include <igl/readTGF.h>

#include <igl/heat_geodesics.h>
#include <igl/avg_edge_length.h>

#include <igl/boundary_conditions.h>
#include <igl/bbw.h>

const std::string MODELNAME = "elephant"; // elephant smpl_50004_punching
const std::string ASSET_PATH = std::string("/Users/bartu/Documents/Github/Spring-Decomp/data/") + MODELNAME + std::string("/");

using namespace Eigen;

// Input polygon
MatrixXd V;
MatrixXi F;
MatrixXd B;

// Tetrahedralized interior
MatrixXd TV;
MatrixXi TT;
MatrixXi TF;

MatrixXd W,C;
MatrixXi BE;

double t = 0;
int n_verts_rendered = 0;
int n_handles;
int selected = 10; // bone weight visulize
double exponent = 4.0; // Euclidean weight decay rate (use >1.0 for exponential decay)
/// -------------------------------------------------------------------------------------------
/// KEY Functions
/// -------------------------------------------------------------------------------------------

void update_weight_colors(igl::opengl::glfw::Viewer& viewer, std::vector<int> s){
    
    MatrixXd W_temp(n_verts_rendered, W.cols());
    for (unsigned i=0; i< s.size();++i)
    {
        //W_temp.row(i) << W.row(i);
        
        W_temp.row(i*4+0) = W.row(TT(s[i],0));
        W_temp.row(i*4+1) = W.row(TT(s[i],1));
        W_temp.row(i*4+2) = W.row(TT(s[i],2));
        W_temp.row(i*4+3) = W.row(TT(s[i],3));
    }
    
    viewer.data().set_data(W_temp.col(selected)); // For weight colors
    return;
}

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  using namespace std;
  using namespace Eigen;
  
  
    
  if( (key >= '1' && key <= '9') ||  key == 'e' || key == 'E' || key == 'r' || key == 'R' )
  {
    t = double((key - '1')+1) / 9.0;
      
    VectorXd v = B.col(2).array() - B.col(2).minCoeff();
    v /= v.col(0).maxCoeff();
      
    vector<int> s;
      
    for (unsigned i=0; i<v.size();++i)
          if (v(i) < t)
              s.push_back(i);
    
      
    n_verts_rendered = int(s.size()) * 4;
    
    MatrixXd V_temp(s.size()*4,3);
    MatrixXi F_temp(s.size()*4,3);
    
    for (unsigned i=0; i<s.size();++i)
    {
      V_temp.row(i*4+0) = TV.row(TT(s[i],0));
      V_temp.row(i*4+1) = TV.row(TT(s[i],1));
      V_temp.row(i*4+2) = TV.row(TT(s[i],2));
      V_temp.row(i*4+3) = TV.row(TT(s[i],3));
      F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
      F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
      F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
      F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
    }

    viewer.data().clear();
    viewer.data().set_mesh(V_temp,F_temp);
    viewer.data().set_face_based(true);
      
       
    update_weight_colors(viewer, s);
  }
    //std::cout << ">>> Key " << key << " pressed" <<std::endl;
    if (key == 'e' || key == 'E' )
    {
        std::cout << "Selected bone: " << selected << "\n";
        selected++;
        if(selected >= n_handles)
            selected = 0;
        
        
        //update_weight_colors(viewer, selected);
        //viewer.data().set_data(W.col(selected));
    }
    if (key == 'r' || key == 'R')
    {
        std::cout << "Selected bone: " << selected << "\n";
        selected--;
        if(selected < 0)
            selected = n_handles - 1;
        selected = std::min(std::max(selected,0),(int)W.cols()-1);
        
        //update_weight_colors(viewer, selected);
        //viewer.data().set_data(W.col(selected));
    }
  
    
    


  return false;
}

/// -------------------------------------------------------------------------------------------
/// UTILS
/// -------------------------------------------------------------------------------------------
double get_distance_squared(Vector3d a, Vector3d b) {
    Vector3d diff_vec = a - b;
    double dist = diff_vec[0]*diff_vec[0] + diff_vec[1]*diff_vec[1] + diff_vec[2]*diff_vec[2];
    return dist;
}

int get_closest_vert_index(Vector3d source_point, const MatrixXd& vertices){
    assert(vertices.cols() == 3); // Assuming vertices shape is (n_verts, 3)
    double min_dist = 99999999;
    int min_dist_idx = -1;
    for(int i = 0; i < vertices.rows(); i++){
        
        Vector3d target_point = vertices.row(i);
        double dist = get_distance_squared(source_point, target_point);
        
        if(dist < min_dist){
            min_dist = dist;
            min_dist_idx = i;
        }
    }
    return min_dist_idx;
}


VectorXd get_handle_vert_dists(Vector3d source_point, const MatrixXd& vertices, bool inverse, double exponential=1.0){
    assert(vertices.cols() == 3); // Assuming vertices shape is (n_verts, 3)
    VectorXd dists(vertices.rows());
    for(int i = 0; i < vertices.rows(); i++){
        
        Vector3d target_point = vertices.row(i);
        double dist_sqr = get_distance_squared(source_point, target_point);
        double dist = sqrt(dist_sqr);
        
        if(inverse)
            dist = 1.0 / pow(dist, exponential);
        
        dists.row(i) << dist;
    }
    return dists;
}


void set_Euclidean_weights(MatrixXd& W, const MatrixXd& C, const MatrixXd& vertices, double exp){
    assert(C.rows() == W.cols());
    assert(C.cols() == 3);
    assert(vertices.rows() == W.rows());
    assert(vertices.cols() == 3);
    
    int n_handles = C.rows();
    for(int i = 0; i < n_handles; i++){
        std::cout << "> Computing Eucliden dists for handle " << i << "...\n";
        
        W.col(i) = get_handle_vert_dists(C.row(i), vertices, true, exp);
        //std::cout << "weights" << get_handle_vert_dists(C.row(i), vertices, true) << std::endl;
    }
    return;
}
 

template <typename MatrixType>
std::string get_matrix_shape(Eigen::MatrixBase<MatrixType>& M){
    return std::to_string(M.rows()) + "x" + std::to_string(M.cols());
}

/// -------------------------------------------------------------------------------------------
/// MAIN
/// -------------------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a surface mesh
  //igl::readOFF(TUTORIAL_SHARED_PATH  "/fertility.off",V,F);
  igl::readOBJ(ASSET_PATH + MODELNAME + ".obj",V,F);

  // Tetrahedralize the interior
  igl::copyleft::tetgen::tetrahedralize(V,F,"pq1.414Y", TV,TT,TF);// pq1.2

  // Compute barycenters
  igl::barycenter(TV,TT,B);

  // Write tet mesh
    igl::writeMESH(ASSET_PATH + MODELNAME + ".mesh", TV, TT, TF);
    int n_tet_verts = TV.rows();
    std::cout << ">>>> INFO: Tetrahedral vertices : " << n_tet_verts << std::endl;

  // read tgf
    if(!igl::readTGF(ASSET_PATH + MODELNAME + ".tgf",C,BE)){
        return EXIT_FAILURE;
    }
    n_handles = C.rows();
    std::cout << ">>>> INFO: Handles : " << n_handles << std::endl;
    std::cout << ">>>> INFO: Bone edges of size : " << BE.rows() << std::endl;
    
    
  // weights -----------------------------------
    W.resize(n_tet_verts, n_handles);
    n_verts_rendered = n_tet_verts;
    set_Euclidean_weights(W,C,TV, exponent);
    
    /*
    VectorXi b;
    MatrixXd bc;
    igl::boundary_conditions(TV,TT,C,VectorXi(),BE,MatrixXi(),MatrixXi(),b,bc);
    
    igl::BBWData bbw_data;
    // only a few iterations for sake of demo
    bbw_data.active_set_params.max_iter = 5;
    bbw_data.verbosity = 2;
    if(!igl::bbw(TV,TT,b,bc,bbw_data,W))
    {
        return EXIT_FAILURE;
    }
    std::cout << ">>>> INFO: Computed BBW.\n";
    */
    // ------------
    
    /*
    // Prepare matrix to hold handle-vertex binding weights
    W.resize(n_tet_verts, n_handles);

    // Precomputation for heat geodesics
    double t = std::pow(igl::avg_edge_length(TV,TF),2);
    std::cout << " >> Found average edge length " << t << ".\n";
    
    igl::HeatGeodesicsData<double> data;
    if(!igl::heat_geodesics_precompute(TV,TF,t,data)){
        std::cerr<<"Error: heat_geodesics_precompute failed."<<std::endl;
        exit(EXIT_FAILURE);
    };
    std::cout << ">> Precomputed heat geodesics data.\n";
    
    // Loop over handles
    for(int i = 0; i < n_handles; i++){
        std::cout << "> Computing heat geodesics for handle " << i << "...\n";
        
        // Step 1 - Get the closest vertex to handle
        int closest_idx = get_closest_vert_index(C.row(i), TV);
        std::cout << "> Found closest vertex at index " << closest_idx << ".\n";
        
        // Step 2 - Compute heat geodesic distances D
        Eigen::VectorXd D;
        Eigen::VectorXi gamma (1,1);
        gamma << closest_idx;
        igl::heat_geodesics_solve(data,gamma,D);
        std::cout << "> Computed heat geodesics of size " << get_matrix_shape(D) << ".\n";
        
        
        // Step 3 - Use inverse D to assign weights
    }
    */
    
    // std::cout << ">>>> INFO: Weights of size " << get_matrix_shape(W) << std::endl;
    
    // Normalize weights to sum to one
    //std::cout << W.row(0) << "--\n";
    W  = (W.array().colwise() / W.array().rowwise().sum()).eval();
    //std::cout << W.row(0);
    igl::writeDMAT(ASSET_PATH + MODELNAME + "_tet.dmat", W);
    
    // Plot the generated mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = &key_down;
    //key_down(viewer,'9',0);
    
    viewer.data().set_mesh(TV, TF);
    //std::cout << W.col(selected);
    viewer.data().set_data(W.col(selected));
    //std::cout << "\nCurrent bone weights: " << W.col(selected);
    viewer.launch();
}
