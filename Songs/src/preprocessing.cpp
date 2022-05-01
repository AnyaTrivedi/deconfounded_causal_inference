#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <utility>
#include <set>
#include <cmath>
#include <filesystem>

int main()
{
    std::string filePath = "/Users/utkarsh/GitHubCodeRepositories/deconfounded_causal_inference/Songs/datasets/train_0.txt";
    std::ifstream fs(filePath);
    if(fs.is_open() == false)
    {
        std::cout << "File Reading failed" << std::endl;
        return 1;
    }

    std::string currLine;

    unsigned long maxUserId = 0;
    unsigned long minUserId = 0;
    unsigned long maxSongId = 0;
    unsigned long minSongId = 0; 
    //std::getline(fs,currLine);
    std::vector<int> indices;
    //std::vector<std::vector<unsigned long> > allData;

    //std::unordered_map<unsigned long, std::vector<std::pair<unsigned long, int> > > userTo_Data;

    std::unordered_map<unsigned long, std::unordered_map<unsigned long, int> > userTo_DataTable; //!Data Table is basically table between song id and rating

    int rowsCount = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while(std::getline(fs,currLine))    
    {
        indices.clear();
        int count = currLine.size();
        for(int i = 0; i < count; ++i)
        {
            if(currLine[i] == '\t')
            {
                indices.push_back(i);
            }
        }

        if(indices.size() != 2)
            continue;

        int startIndex = 0;
//        std::vector<std::string> subs;
        std::vector<unsigned long> subs;        
        for(int index = 0; index < indices.size(); ++index)
        {
            int endIndex = indices[index];
            std::string sub = currLine.substr(startIndex,endIndex - startIndex);
            subs.push_back(std::stoul(sub));
            startIndex = endIndex + 1;
        }
        std::string sub = currLine.substr(startIndex);
        subs.push_back(std::stoul(sub));
        //allData.push_back(subs);

        if(subs.size() != 3)
            continue;

        rowsCount++;

        maxUserId = std::max(maxUserId, subs[0]);
        minUserId = std::min(minUserId, subs[0]);

        maxSongId = std::max(maxSongId, subs[1]);
        minSongId = std::min(minSongId, subs[1]);


        if(userTo_DataTable.find(subs[0]) == userTo_DataTable.end()) //!User not present
        {
            //!Assuming song id rating is unique
            std::unordered_map<unsigned long, int> data;
            //std::vector<std::pair<unsigned long, int> > data;
            //data.push_back(std::make_pair(subs[1], subs[2]));
            //userTo_Data.insert({subs[0], data});

            data.insert({subs[1], subs[2]});
            userTo_DataTable.insert({subs[0], data});
        }
        else    //!User present
        {
            //std::vector<std::pair<unsigned long, int> > &data = userTo_Data[subs[0]];
            //data.push_back(std::make_pair(subs[1], subs[2]));
            std::unordered_map<unsigned long, int> &data = userTo_DataTable[subs[0]];
            data.insert({subs[1], subs[2]});
        }    
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto int_s = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Reading time is " << int_s.count() << " seconds" << std::endl;

    //std::cout << "Numer of rows: " << rowsCount << std::endl;
    std::cout << "Unique Users Count: " << userTo_DataTable.size() << std::endl;

    std::cout << "Max User Id: " << maxUserId << "  Min User Id: " << minUserId << std::endl;
    std::cout << "Max Song Id: " << maxSongId << "  Min Song Id: " << minSongId << std::endl;

    std::vector<unsigned long> allUsers;
    for(auto userRatings : userTo_DataTable)
    {
        allUsers.push_back(userRatings.first);
    }

    std::sort(allUsers.begin(), allUsers.end());

    //!Time taken by writing
    start = std::chrono::high_resolution_clock::now();

    size_t allUsersCount = allUsers.size();
    size_t maxCountOneFile = 5000;
    size_t numFiles = std::ceil(allUsersCount/maxCountOneFile);

    std::string path = "/Users/utkarsh/GitHubCodeRepositories/deconfounded_causal_inference/Songs/datasets/";
    unsigned long startUser = 0;
    for(size_t iFile = 0; iFile < numFiles; ++iFile)
    {
        std::cout << "Writing file: " << iFile << std::endl;
        std::string fileName = std::string("train_0_") + std::to_string(iFile) + std::string(".txt");
        std::string outFile = path + fileName;
        std::fstream fileStream(outFile, std::ios::out);

        //!Header Line
        fileStream << "User Id" << ',';
        for(unsigned long iSongId = minSongId; iSongId <= maxSongId; ++iSongId)
        //for(unsigned long iSongId = minSongId; iSongId <= 100; ++iSongId)
        {
            if(iSongId == maxSongId)
            {
                fileStream << iSongId;            
            }
            else
                fileStream << iSongId << ",";        
        }

        bool userDone = false;
        for(unsigned long iUser = startUser; iUser < startUser + maxCountOneFile; ++iUser)
        //for(unsigned long iUser = 0; iUser < 3; ++iUser)
        {
            if(iUser > allUsers.size())
            {
                userDone = true;
                break;
            }

            fileStream << std::endl;
            unsigned int userId = allUsers[iUser];
            fileStream << userId << ',';

            std::unordered_map<unsigned long, int> &songToRating = userTo_DataTable[userId];
            for(unsigned long iSongId = minSongId; iSongId <= maxSongId; ++iSongId)
            //for(unsigned long iSongId = minSongId; iSongId <= 100; ++iSongId)
            {
                if(songToRating.find(iSongId) == songToRating.end())
                {
                    fileStream << 0;
                }
                else
                {
                    fileStream << songToRating[iSongId];
                }
                if(iSongId != maxSongId)
                {
                    fileStream << ",";
                }
            }
        }
        startUser = startUser + maxCountOneFile;
        if(userDone == true)
        {
            break;
        }
    }




    // //!Print all data
    // for(auto userId : allUsers)
    // {
    //     fileStream << userId << ',';
    // for(unsigned long iSongId = minSongId; iSongId <= maxSongId; ++iSongId)
    // //for(int iSongId = minSongId; iSongId <= 136735; ++iSongId)    
    // {
    //     if(iSongId == maxSongId)
    //     {
    //         fileStream << std::to_string(iSongId);
    //     }
    //     else
    //         fileStream << iSongId << ",";        
    // }
    // fileStream << std::endl;        
    // }

    end = std::chrono::high_resolution_clock::now();


    int_s = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Writing time is " << int_s.count() << " seconds" << std::endl;



    return 0;       
}