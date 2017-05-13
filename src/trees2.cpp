#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

struct node{
    int id;
    node* parent;
    vector<node*> children;
    vector<float> weight;
};

void getLeafNodes(node parent, vector<int> &leaf_vec)
{
   if(parent.children.size() == 0){
//      cout << "leaves = " << parent.id << endl;
      leaf_vec.push_back(parent.id);
      return;
   }
   else{
       for(int i = 0; i < parent.children.size(); i++){
            getLeafNodes(*parent.children[i], leaf_vec);
       }
   }
   return;
}

void getTl(node s, vector<float> &subweights, node root){

   if(s.id == root.id){ 
        cout << "Query invalid for root" << endl;
        return;
   }
   node parent = *(s.parent);
   subweights.push_back(root.weight[0]/parent.weight[0]);
//   subweights.push_back(parent.weight[0]);
//   cout << "Weight = " << parent.weight[0] << endl;
//   vector<node*> children = parent.children;
}

void getSubtrees(node parent, vector<vector<int>> &subleaves, vector<float> &subweights, node root){

    vector<node*> children = parent.children;
    
    for(int i = 0; i < children.size(); i++){
       vector<int> leaf_vec;
       getLeafNodes(*children[i], leaf_vec);
       subleaves.push_back(leaf_vec);
       getTl(*children[i], subweights, root);
       getSubtrees(*children[i], subleaves, subweights, root);
    } 
    return;
}

node getRoot(vector<node> G){
    for(int i = 0; i < G.size(); i++){
        if(G[i].parent == NULL)
            return G[i];
    }
}

int get_m(vector<node> G){

    node root = getRoot(G);
    int m = 0;

    node next_parent = root;
    while(true){
        if(next_parent.children.size() == 0)
            break;
        else
            next_parent = *next_parent.children[0];
        m += 1;
    }    
    return m;
}

int main()
{
    cout << "Hello World!" << endl;

    ifstream treefile("/home/pankaj/SubmodularInference/data/input/tests/trees/tree_stereo_potts.txt");
    string s;

    int nV;
    getline(treefile, s);
    istringstream ss(s);
    ss >> nV;
    cout << "Total number of vertices = " << nV << endl;
    
    vector<node> G(nV);
    for(int i = 0; i < G.size(); i++)
        G[i].id = i;

    int parent_id, child_id;
    float weight;

    while(getline(treefile, s)){
        istringstream ss(s);
    //    cout << s << endl;
        ss >> parent_id;
        ss >> weight;
//        cout << "Parent id = " << parent_id << "        weight = " << weight << endl;
        while(ss >> child_id){
 //           cout << "Child id = " << child_id << endl;
            G[parent_id].children.push_back(&G[child_id]);
            G[parent_id].weight.push_back(weight);
            G[child_id].parent = &G[parent_id];
        }
    }


//    int node_num = 3;
//    vector<node*> a = G[node_num].children;
//    if(a.size() == 0)
//        cout << "Is null" << endl;
//    vector<float> b = G[node_num].weight;
//    for(int i = 0; i < a.size(); i++)
//        cout << "Children of node " + to_string(node_num) + " = " << (*a[i]).id << " weight = " << b[i] << endl;
    vector<vector<int>> subleaves;
    vector<float> subweights;
    node root = getRoot(G);
    vector<float> x = G[root.id].weight;
//    cout << "Root weights = " << endl;
//   for(int i = 0; i < x.size(); i++){
//        cout << "i = " << i << "        " << x[i] << endl;
//    }
    cout << "Root = " << root.id << endl;
    getSubtrees(root, subleaves, subweights, root);
    vector<vector<int>> binaryLeaves;
   for(int i = 0; i < subleaves.size(); i++)
   {
           vector<int> leaves(21, 0);
           cout << "New subtree:" << endl;
           for(int j = 0; j < subleaves[i].size(); j++){
                leaves[subleaves[i][j]] = 1;
               cout << subleaves[i][j] << " ";
           }
            cout << endl; 
           binaryLeaves.push_back(leaves);
            cout << "factor = " << subweights[i] << endl;
   }
   cout << endl;
//    for(int i = 0; i < subleaves.size(); i++){
//           for(int j = 0; j < 21; j++)
//                cout << binaryLeaves[i][j] << " ";
//            cout << endl;                
//    }

//    cout << "m = " << get_m(G) << endl;
    return 0;
}
