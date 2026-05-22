#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "ttt_interfaces/msg/game_snapshot.hpp"
#include "ttt_interfaces/msg/turn_plan.hpp"
#include "ttt_interfaces/msg/workspace_layout.hpp"
#include "ttt_interfaces/srv/plan_turn.hpp"
#include "ttt_interfaces/srv/register_player.hpp"

using namespace std::chrono_literals;

namespace {

std::vector<std::string> panda_joint_names() {
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
          "panda_joint5", "panda_joint6", "panda_joint7"};
}

const std::vector<double> kHomePositions = {0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398};

const std::vector<std::pair<uint8_t, uint8_t>> kScriptedMoves = {
    {6, 2},
    {7, 4},
    {8, 6},
};

// link8 orientation quaternion (x, y, z, w) for gripper pointing down
constexpr double kLink8QuatX = 0.9238795325112867;
constexpr double kLink8QuatY = -0.3826834323650898;
constexpr double kLink8QuatZ = 0.0;
constexpr double kLink8QuatW = 0.0;
// Fixed Z offset from panda_link8 to panda_hand_tcp
constexpr double kLink8TcpZOffset = 0.1034;

geometry_msgs::msg::Pose link8_pose_from_tcp_target(double x, double y, double z) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z + kLink8TcpZOffset;
  pose.orientation.x = kLink8QuatX;
  pose.orientation.y = kLink8QuatY;
  pose.orientation.z = kLink8QuatZ;
  pose.orientation.w = kLink8QuatW;
  return pose;
}

trajectory_msgs::msg::JointTrajectoryPoint make_point(
    const std::vector<double> &positions,
    double time_sec) {
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = positions;
  const auto whole_seconds = static_cast<int32_t>(std::floor(time_sec));
  point.time_from_start.sec = whole_seconds;
  point.time_from_start.nanosec =
      static_cast<uint32_t>((time_sec - static_cast<double>(whole_seconds)) * 1e9);
  return point;
}

moveit_msgs::msg::RobotTrajectory make_three_point_trajectory(
    const std::vector<double> &start_positions,
    const std::vector<double> &end_positions,
    double end_time_sec) {
  moveit_msgs::msg::RobotTrajectory trajectory;
  trajectory.joint_trajectory.joint_names = panda_joint_names();

  const std::vector<double> midpoint = [&]() {
    std::vector<double> result;
    result.reserve(start_positions.size());
    for (size_t index = 0; index < start_positions.size(); ++index) {
      result.push_back((start_positions[index] + end_positions[index]) * 0.5);
    }
    return result;
  }();

  trajectory.joint_trajectory.points.push_back(make_point(start_positions, 0.0));
  trajectory.joint_trajectory.points.push_back(make_point(midpoint, end_time_sec * 0.5));
  trajectory.joint_trajectory.points.push_back(make_point(end_positions, end_time_sec));
  return trajectory;
}


// I HAVE TAKEN REFERENCE OF THIS FROM A YOUTUBE VIDEO
// Source: GeeksforGeeks - Finding optimal move in Tic-Tac-Toe using Minimax
// AND THIS: https://www.youtube.com/watch?v=StXYf5xO-kw&t=2s
// PLEASE DONT BLAME ME I DONT KNOW HOW TO WRITE MINMAX ALGOS
// I HOPE IT WILL WORK
// I AM HONEST THAT I DIDNT WRITE THIS CODE MYSELF, I JUST TRIED TO WRITE WHATEVER I CAN
// =========================================================================
// STEP 1: THE EVALUATOR
// Checks the 1D array to see if someone got 3 in a row.
// Returns +10 if we win, -10 if opponent wins, 0 if no one has won yet.

int evaluate_board(const std::array<uint8_t, 9>& board, uint8_t me, uint8_t opp) {
  // We define the 8 possible winning lines using 0-8 indices.
  // Rows: {0,1,2}, {3,4,5}, {6,7,8} | Cols: {0,3,6}, {1,4,7}, {2,5,8} | Diags: {0,4,8}, {2,4,6}
  const int win_lines[8][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
      {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
      {0, 4, 8}, {2, 4, 6}
  };

  for (int i = 0; i < 8; i++) {
    if (board[win_lines[i][0]] != 0 && 
        board[win_lines[i][0]] == board[win_lines[i][1]] && 
        board[win_lines[i][1]] == board[win_lines[i][2]]) {
      
      if (board[win_lines[i][0]] == me) return 10;
      else if (board[win_lines[i][0]] == opp) return -10;
    }
  }
  return 0; // Nobody won yet
}

// =========================================================================
// STEP 2: THE RECURSIVE MINIMAX ALGORITHM
// Plays out every possible future move to the end of the game.
int minimax(std::array<uint8_t, 9> board, int depth, bool is_max, uint8_t me, uint8_t opp) {
  int score = evaluate_board(board, me, opp);
  
  // If a timeline ends in victory or defeat, return the score
  if (score == 10) return score - depth; // Subtract depth so it prefers faster wins
  if (score == -10) return score + depth; // Add depth so it delays losses as long as possible

  // Check if the board is full (a tie)
  bool moves_left = false;
  for (int i = 0; i < 9; i++) {
    if (board[i] == 0) moves_left = true; // 0 means EMPTY
  }
  if (!moves_left) return 0;

  if (is_max) {
    int best = -1000;
    for (int i = 0; i < 9; i++) {
      if (board[i] == 0) { // If cell is empty
        board[i] = me; // Try playing our piece
        best = std::max(best, minimax(board, depth + 1, !is_max, me, opp));
        board[i] = 0;  // Undo the move
      }
    }
    return best;
  } else {
    int best = 1000;
    for (int i = 0; i < 9; i++) {
      if (board[i] == 0) { // If cell is empty
        board[i] = opp; // Try playing opponent's piece
        best = std::min(best, minimax(board, depth + 1, !is_max, me, opp));
        board[i] = 0;   // Undo the move
      }
    }
    return best;
  }
}

// =========================================================================
// STEP 3: THE DECISION MAKER
// Loops through currently empty cells and uses Minimax to pick the best one.
uint8_t find_best_move(std::array<uint8_t, 9> board, uint8_t me, uint8_t opp) {
  int best_val = -1000;
  uint8_t best_move = 255; // Default to an invalid ID

  for (int i = 0; i < 9; i++) {
    if (board[i] == 0) { // Found an empty cell
      board[i] = me; // Temporarily place our marker
      int move_val = minimax(board, 0, false, me, opp); // Calculate how good this move is
      board[i] = 0; // Undo the move

      if (move_val > best_val) {
        best_move = i;
        best_val = move_val;
      }
    }
  }
  return best_move;
}
// END OF MINMAX --------------------------------



}  // namespace

class StudentPlayerNode : public rclcpp::Node {
 public:
  StudentPlayerNode() : Node("student_player") {
    this->declare_parameter<std::string>("player_name", this->get_name());
    this->declare_parameter<std::string>("plan_turn_service", "/student_player/plan_turn");

    player_name_ = this->get_parameter("player_name").as_string();
    plan_turn_service_ = this->get_parameter("plan_turn_service").as_string();

    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    register_client_ =
        this->create_client<ttt_interfaces::srv::RegisterPlayer>("/ttt/register_player");

    ik_client_ = this->create_client<moveit_msgs::srv::GetPositionIK>(
        "/compute_ik",
        rmw_qos_profile_services_default,
        cb_group_);

    plan_turn_service_server_ = this->create_service<ttt_interfaces::srv::PlanTurn>(
        plan_turn_service_,
        std::bind(&StudentPlayerNode::handle_plan_turn, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default,
        cb_group_);

    register_timer_ =
        this->create_wall_timer(500ms, std::bind(&StudentPlayerNode::try_register, this));
  }

 private:
  int registration_delay_ticks_{0};
  void try_register() {
    // Apparently my student player keeps registering as the dummy too quickly i hope this is acceptable
    if (registration_delay_ticks_ < 6) {
      registration_delay_ticks_++;
      return;
    }

    if (registered_ || registration_in_flight_) {
      return;
    }
    if (!register_client_->wait_for_service(100ms)) {
      return;
    }

    auto request = std::make_shared<ttt_interfaces::srv::RegisterPlayer::Request>();
    request->player_name = player_name_;
    request->service_name = plan_turn_service_;
    registration_in_flight_ = true;

    register_client_->async_send_request(
        request,
        [this](rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedFuture future) {
          registration_in_flight_ = false;
          try {
            const auto response = future.get();
            if (!response->success) {
              RCLCPP_WARN(this->get_logger(), "Registration rejected: %s",
                          response->message.c_str());
              return;
            }
            registered_ = true;
            player_id_ = response->assigned_player_id;
            RCLCPP_INFO(this->get_logger(), "Registered as player_%u.", player_id_);
            register_timer_->cancel();
          } catch (const std::exception &exc) {
            RCLCPP_ERROR(this->get_logger(), "Registration failed: %s", exc.what());
          }
        });
  }

  // ----------------------------------------------------------------
  // IK helpers
  // ----------------------------------------------------------------

  std::optional<std::vector<double>> compute_ik(
      const geometry_msgs::msg::Pose &target_pose,
      const std::vector<double> &seed_positions) {
    (void)target_pose;
    (void)seed_positions;

    // TODO(student): Call the MoveIt `/compute_ik` service here.
    // Suggested steps:
    // 1. Create a `moveit_msgs::srv::GetPositionIK::Request`.
    // 2. Set `group_name = "panda_arm"`.
    // 3. Fill the seed joint state with the provided `seed_positions`.
    // 4. Set the target pose in frame `panda_link0`.
    // 5. Send the request through `ik_client_` and wait for the response.
    // 6. Extract the 7 Panda arm joints from the solution and return them.
    // 7. Return `std::nullopt` if IK times out or fails.
    //
    // The dummy return below keeps the starter code buildable, but it does not
    // solve IK. Students should replace it with a real implementation.

    // 1. Create a `moveit_msgs::srv::GetPositionIK::Request`.
    // 2. Set `group_name = "panda_arm"`.
    // 3. Fill the seed joint state with the provided `seed_positions`.
    // 4. Set the target pose in frame `panda_link0`.
    // 5. Send the request through `ik_client_` and wait for the response.
    // from Lab 12 exercise1
    auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();

    request->ik_request.group_name = "panda_arm";
    request->ik_request.ik_link_name = "panda_link8"; //Target link

    request->ik_request.robot_state.joint_state.name = panda_joint_names(); // Load joints 1-7

    request->ik_request.robot_state.joint_state.position = seed_positions; // seed pos is the starting default position
    request->ik_request.pose_stamped.header.frame_id = "panda_link0"; // Set reference at link0 robot's base
    request->ik_request.pose_stamped.pose = target_pose; // Set target pose cords
    request->ik_request.timeout.sec = 2; // 2 seconds thinking time
    request->ik_request.avoid_collisions = true; // This is just no shit

    // 6. Extract the 7 Panda arm joints from the solution and return them.
    // 7. Return `std::nullopt` if IK times out or fails.
    auto future = ik_client_->async_send_request(request); 

    /*
    // THi is straight from lab
    // Pause for 3s and take the response
    const auto result = rclcpp::spin_until_future_complete(
        this->get_node_base_interface(), future, 3s);
    
    // Just anthing that is NOT SUCCESS
    if (result != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Timed out while waiting for /compute_ik.");
      return std::nullopt; // std::nullopt means "fail, return nothing" apparently I leant 
    }
    */

    // Wait patiently for the MultiThreadedExecutor to process the response in the background
    auto status = future.wait_for(3s);
    
    // If the status isn't ready after 3 seconds, it timed out
    if (status != std::future_status::ready) {
      RCLCPP_ERROR(this->get_logger(), "Timed out while waiting for /compute_ik.");
      return std::nullopt;
    }

    // response is in a FUTURE object i guess
    const auto response = future.get();
    // Check for errors inside the response
    if (response->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "IK failed with MoveIt error code %d.", response->error_code.val);
      return std::nullopt;
    }

    // EXTRACTION ---------------
    // FROM extract_panda_solution() LAB12 ----------
    std::vector<double> solution;
    solution.reserve(panda_joint_names().size());

    const auto names = panda_joint_names();

    // short loop through our master list
    for (const auto& joint_name : names) {
      const auto& response_names = response->solution.joint_state.name;
      
      auto iter = std::find(response_names.begin(), response_names.end(), joint_name);

      // If joint is missing????? how dat happens
      if (iter == response_names.end()) {
        RCLCPP_ERROR(this->get_logger(), "IK solution is missing %s.", joint_name.c_str());
        return std::nullopt;
      }
      const size_t index = std::distance(response_names.begin(), iter);

      // push onto Cleaned up list 
      solution.push_back(response->solution.joint_state.position[index]);
    }
    // END OF extract_panda_solution() ---------------

    // Print IK solution to console for MY sanity
    RCLCPP_INFO(this->get_logger(), "IK succeeded. Panda arm joint solution:");
    for (size_t index = 0; index < names.size(); ++index) {
      RCLCPP_INFO(this->get_logger(), "  %s = %.6f", names[index].c_str(), solution[index]);
    }

    return solution;

  }

  // ----------------------------------------------------------------
  // Turn planning
  // ----------------------------------------------------------------

  void handle_plan_turn(
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> request,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> response) {
    if (request->player_id != player_id_) {
      response->accepted = false;
      response->message = "Plan request does not match registered player id.";
      return;
    }

    // TODO(student): Implement your turn-planning logic here.
    // Suggested structure:
    // 1. Choose a legal `(piece_id, cell_id)` pair from `request->snapshot`.
    // 2. Look up the current pose of the chosen stock piece.
    // 3. Look up the target board cell pose from `request->layout.cell_poses`.
    // 4. Convert those TCP targets into `panda_link8` poses using
    //    `link8_pose_from_tcp_target(...)`.
    // 5. Call `compute_ik(...)` for the pick target and place target.
    // 6. Build the four required trajectories:
    //      - home_to_pick
    //      - pick_to_home
    //      - home_to_place
    //      - place_to_home
    // 7. Fill `response->plan` and set `response->accepted = true` on success.
    //
    // The fallback below intentionally rejects every turn. This keeps the
    // starter repository buildable while making it clear that students must
    // implement their own planner.

    // CHOOSE A CELL
    // MINMAX ALGO --------------------
    // The snapshot already gives us a list of empty cells. We just grab the first one.
    if (request->snapshot.legal_actions.empty()) {
      response->accepted = false;
      response->message = "No legal moves left on the board.";
      return;
    }

    
    // We need to figure out which marker number we are on the board (1 or 2).
    // player_id_ is 0 or 1. The GameSnapshot board uses 1 for Player 0, and 2 for Player 1.
    // So we just add 1 to our player_id_ to get our board marker, Just trust this process
    uint8_t my_marker = player_id_ + 1;
    uint8_t opp_marker = (player_id_ == 0) ? 2 : 1;

    uint8_t target_cell_id = find_best_move(request->snapshot.board, my_marker, opp_marker);

    if (target_cell_id == 255) {
      target_cell_id = request->snapshot.legal_actions[0];
      RCLCPP_WARN(this->get_logger(), "Minimax failed. Falling back to cell %u", target_cell_id);
    }
    

    // Just first legal move for testing
    //uint8_t target_cell_id = request->snapshot.legal_actions[0];

    // 2. Look up the current pose of the chosen stock piece.
    // Loop through all the pieces on the table to find a blue cube we are allowed to use
    uint8_t target_piece_id = 69;
    // Look thru all pieces
    for (const auto& piece : request->snapshot.pieces) {
      if (piece.owner == player_id_ && piece.available) { // My piece and avaliable piece
        target_piece_id = piece.piece_id; // just grab the first one
        break; // Stop
      }
    }
    // No pieces left
    if (target_piece_id == 69) {
      response->accepted = false;
      response->message = "No more available pieces to move.";
      return;
    }

    // 3. Look up the target board cell pose from `request->layout.cell_poses`
    auto piece_pose = find_piece_pose(request->snapshot, target_piece_id);
    auto cell_pose = request->layout.cell_poses[target_cell_id];

    // 4. Convert those TCP targets into `panda_link8` poses using
    //    `link8_pose_from_tcp_target(...)`.
    auto pickup_pose = link8_pose_from_tcp_target(piece_pose.position.x, piece_pose.position.y, piece_pose.position.z);
    auto place_pose = link8_pose_from_tcp_target(cell_pose.position.x, cell_pose.position.y, cell_pose.position.z);

    // 5. Call `compute_ik(...)` for the pick target and place target.
    auto pickup_ik = compute_ik(pickup_pose, request->layout.home_joint_state.position);
    auto place_ik = compute_ik(place_pose, request->layout.home_joint_state.position);

    // If MY IK solver failed (returned std::nullopt), we have to abort.
    if (!pickup_ik.has_value()) {
      response->accepted = false;
      response->message = "Failed to compute IK for pick targets.";
      return;
    }
    if (!place_ik.has_value()) {
      response->accepted = false;
      response->message = "Failed to compute IK for place targets.";
      return;
    }

    // 6. Build the four required trajectories:
    //      - home_to_pick
    //      - pick_to_home
    //      - home_to_place
    //      - place_to_home
    // Lab12
    auto home_to_pick = make_three_point_trajectory(request->layout.home_joint_state.position, *pickup_ik, 2.0);
    auto pick_to_home = make_three_point_trajectory(*pickup_ik, request->layout.home_joint_state.position, 2.0);
    auto home_to_place = make_three_point_trajectory(request->layout.home_joint_state.position, *place_ik, 2.0);
    auto place_to_home = make_three_point_trajectory(*place_ik, request->layout.home_joint_state.position, 2.0);


    // 7. Fill `response->plan` and set `response->accepted = true` on success.
    response->plan.match_id = request->snapshot.match_id;
    response->plan.turn_index = request->snapshot.turn_index;
    response->plan.player_id = player_id_;
    response->plan.piece_id = target_piece_id;
    response->plan.cell_id = target_cell_id;

    response->plan.home_to_pick = home_to_pick;
    response->plan.pick_to_home = pick_to_home;
    response->plan.home_to_place = home_to_place;
    response->plan.place_to_home = place_to_home;

    // Logger!
    RCLCPP_INFO(this->get_logger(), "Planned turn for piece %u to cell %u.", target_piece_id, target_cell_id);
    RCLCPP_INFO(this->get_logger(), "home_to_pick trajectory has %zu points.", home_to_pick.joint_trajectory.points.size());
    RCLCPP_INFO(this->get_logger(), "pick_to_home trajectory has %zu points.", pick_to_home.joint_trajectory.points.size());
    RCLCPP_INFO(this->get_logger(), "home_to_place trajectory has %zu points.", home_to_place.joint_trajectory.points.size());
    RCLCPP_INFO(this->get_logger(), "place_to_home trajectory has %zu points.", place_to_home.joint_trajectory.points.size());
    RCLCPP_INFO(this->get_logger(), "Turn planned successfully, accepting.");
    
    
    response->accepted = true;
    response->message = "Turn planned successfully.";
  }


  static geometry_msgs::msg::Pose find_piece_pose(
      const ttt_interfaces::msg::GameSnapshot &snapshot,
      uint8_t piece_id) {
    // TODO(student): Search `snapshot.pieces` for the requested `piece_id` and
    // return its pose. You may choose to throw an exception or return a
    // fallback pose if the piece is missing.

    // Loop through snapshot piece
    for (const auto& piece : snapshot.pieces) {
  
      if (piece.piece_id == piece_id) {
        // Return the pose
        return piece.pose;
      }
    }

    // Dummy fallback to keep the starter code compilable.
    geometry_msgs::msg::Pose fallback;
    fallback.orientation.w = 1.0;
    return fallback;
  }

  std::string player_name_;
  std::string plan_turn_service_;
  bool registered_{false};
  bool registration_in_flight_{false};
  uint8_t player_id_{255};

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedPtr register_client_;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
  rclcpp::Service<ttt_interfaces::srv::PlanTurn>::SharedPtr plan_turn_service_server_;
  rclcpp::TimerBase::SharedPtr register_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StudentPlayerNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
