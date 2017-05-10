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

int main()
{
    cout << "Hello World!" << endl;

    ifstream treefile("/home/pankaj/SubmodularInference/data/input/tests/trees/tree2.txt");
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
        cout << "Parent id = " << parent_id << "        weight = " << weight << endl;
        while(ss >> child_id){
            cout << "Child id = " << child_id << endl;
            G[parent_id].children.push_back(&G[child_id]);
            G[parent_id].weight.push_back(weight);
            G[child_id].parent = &G[parent_id];
        }
    }

    int node_num = 3;
    vector<node*> a = G[node_num].children;
    vector<float> b = G[node_num].weight;
    for(int i = 0; i < a.size(); i++)
        cout << "Children of node " + to_string(node_num) + " = " << (*a[i]).id << " weight = " << b[i] << endl;
    return 0;
}
