pragma solidity ^0.8.11;

// SPDX-License-Identifier: MIT
contract ModelTrain {
     
    // train info
    struct TrainInfo {
        address trainer;
        uint dataSize; // the size of training data
        uint version; // the version of this train info
        string modelUpdateHash; // the model update info's file hash in IPFS network
        uint poll; // the number of votes this model update get
    }
    // trainer's detail info
    struct TrainerDetail {
        address trainer;
        bool hasVoted; // whether the trainer give the vote
        address[] candidates; // the candidates this trainer vote
        TrainInfo trainInfo;
    }
    // snapshot at specific epoch 
    struct Snapshot{
        address[] trainers;
        mapping(address => TrainerDetail) details;
        uint version;
        uint nVoteCollect; // numbers of vote this snapshot get
        bool trainFinished; // if locked is true ,then the train process is finished
        bool voteFinished; // if voteFinished is true, then current snapshot is finished,can't update again
    }
    struct TaskSetting{
        string taskDescription; // the description of the federate learning task
        string modelDescription; // the description of the model(e.g. model name,model struct)
        string datasetDescription; // the description of the dataset(e.g. struct of the dataset
    }
    struct TrainSetting{
        uint batchSize; // local training batch size
        string learningRate; // local training learning rate (float)
        uint epochs; // local training epoch step
        uint nTrainer; // the number of client participate in one global model update epoch
        uint nPoll; // the number of polls one trainer has
        uint nClient; // the number of clients participate in
        uint maxVersion; // the max version we train
    }
    struct ClientInfo{
        address addr; // the address of this client
        string datasetDescription; // the dataset description of this client
        string extraDescription; // extra description
    }
    struct Setting{
        TaskSetting task;
        TrainSetting train;
    }
    address public publisher;
    uint public curVersion;
    enum CurState{
        COLLECT_MODEL, //0
        COLLECT_VOTE, //1
        STOP //2
    }
    CurState public curState;
    // global setting of training
    Setting public setting;
    // version => snapshot with given version
    mapping(uint => Snapshot) snapshots;
    // use to record the contribution
    mapping(address => uint) public contributions;

    // use to record the client's info
    mapping(address => bool) public enroll;
    ClientInfo[] public clientInfos;

    event UploadTrainInfo(address _uploader,uint _version);
    event NeedAggregation(uint _version);
    event NeedVote(uint _version);

    constructor(TrainSetting memory _train,TaskSetting memory _task) {
        publisher = msg.sender;
        curVersion = 0; 
        setting.task = _task;
        setting.train = _train;
    }

    // check whether the last version exist trainer
    function exist(address trainer) internal view returns (bool){
        Snapshot storage snapshot = snapshots[curVersion];
        for(uint i=0;i<snapshot.trainers.length;i++){
            if(trainer == snapshot.trainers[i]){
                return true;
            }
        }
        return false;
    }

    // init the federate learning task
    function init(
        TrainSetting memory  _train,
        TaskSetting memory _task,
        string memory _initHash)
        public returns (bool){
        setting.train = _train;
        setting.task = _task;
        initTrainInfo(_initHash);
        return true;
    }

    // init the train info with init model hash _initHash
    function initTrainInfo(string memory _initHash) public returns (bool){
        require(curVersion == 0,"model is already inited");
        uploadTrainInfo(0, _initHash);
        address[] memory votes;
        vote(votes);
        return true;
    }

    // enroll the train task
    function enrollTrain(string memory _datasetDesc,string memory _extraDesc) external returns (bool){
        require(clientInfos.length < setting.train.nClient,"no more client can enroll");
        require(enroll[msg.sender] == false,"client has enrolled");
        ClientInfo memory info = ClientInfo(
            msg.sender,
            _datasetDesc,
            _extraDesc
        );
        clientInfos.push(info);
        enroll[msg.sender] = true;
        return true;
    }

    // get all train infos (local training) within specific version
    function getTrainInfos(uint _version) view public returns (TrainInfo[] memory) {
        require(_version <= curVersion,"invalid version");
        Snapshot storage snapshot = snapshots[_version];
        TrainInfo[] memory infos = new TrainInfo[](snapshot.trainers.length);
        for(uint i=0;i< snapshot.trainers.length; i++){
            address trainer = snapshot.trainers[i];
            infos[i] = snapshot.details[trainer].trainInfo;
        }
        return infos;
    }

    // get all train infos finished within the last version
    function getTrainInfos() view public returns (TrainInfo[]memory){
        if(snapshots[curVersion].trainFinished){
            return getTrainInfos(curVersion);
        }else{
            return getTrainInfos(curVersion-1);
        }
    } 

    // upload train info
    function uploadTrainInfo(uint _version,uint _dataSize,string memory _modelUpdateHash) public returns (bool){
        require(curVersion <= setting.train.maxVersion,"model train finished");
        require(_version == curVersion,"unexpected version");
        require(enroll[msg.sender] == true,"uploader do not enroll the task");
        Snapshot storage snapshot = snapshots[curVersion];
        // new trainer of current version
        require(!snapshot.trainFinished,"current version's local updates collected finished");
        if (!exist(msg.sender)){
            snapshot.trainers.push(msg.sender);    
        }
        emit UploadTrainInfo(msg.sender, curVersion);
        snapshot.details[msg.sender].trainInfo = TrainInfo(msg.sender, _dataSize,_version, _modelUpdateHash,0);
       // train info collect finished
        if(snapshot.trainers.length == setting.train.nTrainer || curVersion == 0){
            snapshot.trainFinished = true;
            emit NeedVote(curVersion);
            curState = CurState.COLLECT_VOTE;
        }
        return true;
    }
    
    function uploadTrainInfo(uint _dataSize,string memory _modelUpdateHash) public returns (bool){
        return uploadTrainInfo(curVersion, _dataSize, _modelUpdateHash);
    }

    // trainer give the poll
    function vote(address[] memory _candidates,uint _version) public returns (bool){
        require(curVersion <= setting.train.maxVersion,"model train finished");
        require(_candidates.length <= setting.train.nPoll,"too many candidates");
        require(_version == curVersion,"unexpected version");
        Snapshot storage snapshot = snapshots[curVersion];
        require(snapshot.trainFinished,"current version's local updates collected do not finished");
        require(exist(msg.sender),"only trainer can vote");
        // the trainer's detail
        TrainerDetail storage detail = snapshot.details[msg.sender];
        require(!detail.hasVoted,"this trainer has voted before");
        for(uint i=0;i< _candidates.length;i++){
            require(exist(_candidates[i]),"candidate do no exist");
        }
        // save the vote record
        snapshot.details[msg.sender].candidates = _candidates;
        snapshot.nVoteCollect ++ ;
        if (snapshot.nVoteCollect == setting.train.nTrainer ||
            snapshot.version == 0 // for the init case
            ){
            // current version's local updates collected finished , need to be aggregated
            emit NeedAggregation(curVersion);
            snapshot.voteFinished = true;
            // create snapshot for new version
            // update the poll for every update model info
            for(uint i=0;i<snapshot.trainers.length;i++){
                address voter = snapshot.trainers[i];
                address[] storage candidates = snapshot.details[voter].candidates;
                for(uint j=0;j<candidates.length;j++){
                    snapshot.details[candidates[j]].trainInfo.poll++;
                }
            }
            // update the contributions
            for(uint i=0;i<snapshot.trainers.length;i++){
                address trainer = snapshot.trainers[i];
                TrainInfo storage trainInfo = snapshot.details[trainer].trainInfo;
                uint f = 1;
                if(3 * (trainInfo.poll+1) >= 2* setting.train.nPoll){
                    f = 2;
                }
                contributions[trainer] += (trainInfo.poll) * (trainInfo.dataSize) * f;
            }
            curVersion ++;
            curState = CurState.COLLECT_MODEL;
            snapshots[curVersion].version = curVersion;
        }
        return true;
    }

     function vote(address[] memory _votes) public returns (bool){
         return vote(_votes,curVersion);
     }
}