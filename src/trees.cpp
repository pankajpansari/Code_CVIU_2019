#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>

using namespace std;

struct node{
    int id;
    node* parent;
    vector<node*> children;
    vector<float> weight;
};

void getLeafNodes(node parent, std::vector<int> &leaf_vec)
{
    using namespace std;
    if(parent.children.size() == 0){
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

void get_node_weight(node s, std::vector<float> &node_weights, node root){
    using namespace std;
    if(s.id == root.id){ //root assumed to have id 0
        return;
    }
    node parent = *(s.parent);
    //       node_weights.push_back(root.weight[0]/parent.weight[0]);
    node_weights.push_back(parent.weight[0]);
}

void getSubtrees(node parent, std::vector<std::vector<int>> &subleaves, std::vector<float> &node_weights, node root){

    using namespace std;
    vector<node*> children = parent.children;

    for(int i = 0; i < children.size(); i++){
        vector<int> leaf_vec;
        getLeafNodes(*children[i], leaf_vec);
        subleaves.push_back(leaf_vec);
        get_node_weight(*children[i], node_weights, root);
        getSubtrees(*children[i], subleaves, node_weights, root);
    } 
    return;
}

node getRoot(std::vector<node> G){
    for(int i = 0; i < G.size(); i++){
        if(G[i].parent == NULL)
            return G[i];
    }
}

void printTree(vector<vector<int>> subleaves, vector<float> node_weights, node root){
    cout << "Root id = " <<  root.id << endl;
   for(int i = 0; i < subleaves.size(); i++){
        cout << "Node " << i << " subleaves" << endl;
        for(int j = 0; j < subleaves[i].size(); j++){
            cout << subleaves[i][j] << " "; 
        }
        cout << endl;
   }  
}

//void getPath(node t, const vector<node> &G, vector<node> &path_nodes){
void getPath(node t, const vector<node> &G, vector<node> &path_nodes){
    //given the tree G, and the leaf node t, get the list of nodes on the path from root to t
//    node parent_node = *t.parent; 
    node parent_node = t;
    while(parent_node.parent != NULL){
        path_nodes.push_back(parent_node);
        parent_node = *parent_node.parent; 
    }
}

int main(){
   
    const string filename = "/home/pankaj/SubmodularInference/data/input/tests/trees/tree2.txt";

    ifstream treefile(filename);
    string s;

    vector<vector<int>> subleaves;
    vector<float> node_weights; 
 
    int nV;
    getline(treefile, s);
    istringstream ss(s);
    ss >> nV;

    vector<node> G(nV);
    for(int i = 0; i < G.size(); i++)
        G[i].id = i;

    int parent_id, child_id;
    float weight;

    while(getline(treefile, s)){
        istringstream ss(s);
        ss >> parent_id;
        ss >> weight;
        while(ss >> child_id){
            G[parent_id].children.push_back(&G[child_id]);
            G[parent_id].weight.push_back(weight);
            G[child_id].parent = &G[parent_id];
        }
    }

    node root = getRoot(G);
    getSubtrees(root, subleaves, node_weights, root);
    printTree(subleaves, node_weights, root);
//    cout << subleaves.size() << endl;

    vector<int> leaf_nodes;
    getLeafNodes(root, leaf_nodes);

//    cout << "Leaves" << endl;
//    for(int j = 0; j < leaf_nodes.size(); j++){
//        cout << leaf_nodes[j] << " "; 
//    }
//    cout << endl;

    vector<node> path_nodes;
    getPath(G[14], G, path_nodes);

    cout << "Path = " << endl;
    for(int i = 0; i < path_nodes.size(); i++)
        cout << path_nodes[i].id << " ";
    return 0;
}
