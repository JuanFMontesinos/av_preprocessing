echo "Building 3DDFA_V2 dependencies..."

echo "Building FaceBoxes..."
cd FaceBoxes
sh ./build_cpu_nms.sh
cd ..

echo "Building Sim3DR..."
cd Sim3DR
sh ./build_sim3dr.sh
cd ..

echo "Building utils/asset..."
cd utils/asset
gcc -shared -Wall -O3 render.c -o render.so -fPIC
cd ../..