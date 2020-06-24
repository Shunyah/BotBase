
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ViZDoom.h>
#include <chrono>
#include <thread>
#include <opencv2/core/mat.hpp>

double total_reward = 0;

using namespace vizdoom;
using namespace cv;
using namespace std;

std::string path = "E:\\practice\\vizdoom";
auto game = std::make_unique <vizdoom::DoomGame>();
auto screenBuff = cv::Mat(480, 640, CV_8UC3);
const unsigned int sleepTime = 1000 / vizdoom::DEFAULT_TICRATE;

const std::vector<double> actions1[3] = {
	{ 1, 0, 0}, // left
	{ 0, 1, 0}, // right
	{ 0, 0, 1}, // âóä
};

const std::vector<double> actions2[4] = {
	{ 1, 0, 0, 0 }, // left
	{ 0, 1, 0, 0 }, // right
	{ 0, 0, 1, 0 }, // âóä 
	{ 0, 0, 0, 1 }, // shoot
};


void game_init() {
	game->setViZDoomPath("../vizdoom/vizdoom");
	game->setDoomGamePath("../vizdoom/freedoom2.wad");
	game->loadConfig("../vizdoom/scenarios/task1.cfg"); 
	game->setScreenResolution(RES_640X480); 
	game->setLabelsBufferEnabled(1); 
	game->setWindowVisible(1);
	game->setRenderWeapon(1); 
	game->setRenderHud(1);
	game->init();
}

int find(cv::Mat screen) {
	CvPoint sum = cvPoint(0, 0);
	for (int x = 100; x < (&screen)->cols; ++x)
		for (int y = 200; y < 250; ++y)
			if (screen.at<unsigned char>(y, x) == 255) {
				sum.x += x + 15;
				return sum.x;
			}
}

void kill1(int x) {
	if (x > 355)
		game->makeAction(actions1[1]); //right
	else if (x < 285)
		game->makeAction(actions1[0]); //left
	else
		game->makeAction(actions1[2]); //shoot
}

void kill2(int x) {
	if (x > 363)
		game->makeAction(actions2[1]); //right
	else if (x < 278)
		game->makeAction(actions2[0]); //left
	else
		game->makeAction(actions2[3]); //shoot
}

void sleep(size_t time) 
{
	std::this_thread::sleep_for(std::chrono::milliseconds(time));
}

CvPoint find2(cv::Mat matrix) {
	CvPoint sum = cvPoint(0, 0);
	for (int x = 239; x < (&matrix)->cols; x++) {
		for (int y = 201; y < 210; y++) {
			if (matrix.at<unsigned char>(y, x) == 255) {
				sum.x += x + 15;
				//sum.y += y;
				return sum;
			}
		}
	}

}

void RunTask1(int episodes)
{
	game_init();

	auto image = cv::Mat(480, 640, CV_8UC3);
	auto greyscale = cv::Mat(480, 640, CV_8UC1);
	cvNamedWindow("Game");
	cv::moveWindow("Game", 80, 30);
	cvNamedWindow("Greyscale");
	cv::moveWindow("Greyscale", 680, 30);

	for (auto i = 0; i < episodes; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;
		while (!game->isEpisodeFinished())
		{
			const auto& gamestate = game->getState();
			std::memcpy(image.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

			cv::extractChannel(image, greyscale, 1);
			cv:threshold(greyscale, greyscale, 130, 255, cv::THRESH_BINARY);

			imshow("Greyscale", greyscale);
			imshow("Game", image);
			int x = find(greyscale);
			kill1(x);

			char c = cv::waitKey(sleepTime);
			if (c == 27) break;
		}
		std::cout << game->getTotalReward() << std::endl;
		total_reward += game->getTotalReward();
	}
	std::cout << total_reward / episodes << endl;
	cvWaitKey(0);
	cvDestroyAllWindows();
}

void RunTask2(int episodes)
{
	game_init();

	auto image = cv::Mat(480, 640, CV_8UC3);
	auto greyscale = cv::Mat(480, 640, CV_8UC1);
	
	cvNamedWindow("Game");
	cv::moveWindow("Game", 40, 30);
	cvNamedWindow("Greyscale");
	cv::moveWindow("Greyscale", 680, 30);

	size_t sleepTime = 1000 / DEFAULT_TICRATE;

	for (int i = 0; i < episodes; i++) 
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;
		while (!game->isEpisodeFinished()) 
		{
			
			const auto& gamestate = game->getState();
			std::memcpy(image.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());
			
			cv::extractChannel(image, greyscale, 1);
			cv::threshold(greyscale, greyscale, 130, 255, cv::THRESH_BINARY);


			CvPoint w = find2(greyscale);
			kill2(w.x);
			cv::imshow("Game", image);
			cv::imshow("Greyscale", greyscale);
			
			cvWaitKey(sleepTime);
		}
		sleep(sleepTime*10);
		std::cout << "Total reward is: " << game->getTotalReward() << std::endl;
		total_reward += game->getTotalReward();
	}
	std::cout << total_reward / episodes << std::endl;
	cvWaitKey(0);
	cvDestroyAllWindows();
}

void RunTask3(int episodes)
{
	try
	{
		game->loadConfig(path + "\\scenarios\\task3.cfg");

		game->init();

	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	auto greyscale = cv::Mat(480, 640, CV_8UC1);
	auto image = cv::Mat(480, 640, CV_8UC3);
	std::vector<double> actions[5];

	actions[0] = { 1,0,0,0 };
	actions[1] = { 0,1,0,0 };
	actions[2] = { 0,0,1,0 };
	actions[3] = { 0,0,0,1 };
	actions[4] = { 0,0,0,0 };

	cv::Mat clusters;
	for (auto i = 0; i < episodes; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;

		while (!game->isEpisodeFinished())
		{

			const auto& gamestate = game->getState();

			std::memcpy(screenBuff.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

			cv::extractChannel(screenBuff, greyscale, 2);

			cv::threshold(greyscale, greyscale, 200, 255, cv::THRESH_BINARY);

			std::vector<cv::Point2f> field(0);

			for (int w = 0; w < 640; w++)
			{
				for (int x = 0; x < 410; x++)
				{
					if ((int)greyscale.at<unsigned char>(x, w) == 255) {
						field.push_back(cv::Point2f(w, x));

					}
				}

			}
std::vector<cv::Point2f> kit;
			

			if (field.size() > 1) {
				// clusters
				greyscale.convertTo(greyscale, CV_32F);
				cv::kmeans(field, 2, clusters, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS, kit);
				greyscale.convertTo(greyscale, CV_8UC1);

				// closest kit
				int dest1 = (kit[0].x - 320)* (kit[0].x - 320) + (kit[0].y - 410)* (kit[0].y - 410);
				int dest2 = (kit[1].x - 320)* (kit[1].x - 320) + (kit[1].y - 410)* (kit[1].y - 410);

				int kit_w = 0;
				int kit_x = 0;

				if (dest1 < dest2) {
					kit_w = kit[0].x;
					kit_x = kit[0].y;
				}
				else {
					kit_w = kit[1].x;
					kit_x = kit[1].y;
				}
				cv::circle(greyscale, cv::Point(kit_w, kit_x),20, cv::Scalar(200, 100, 100), 5);
				// хотела сделать красиво а получилось как всегда
				for (int i = 0; i < kit.size(); i++)
				{
					cv::Point v = kit[i];

					//cv::circle(greyscale, v, 5, cv::Scalar(0, 0, 255), -1, 8);
					//cv::rectangle(field, cv::Rect(v.x - 25, v.y - 25, 50, 50), cv::Scalar(0, 0, 255));
				}

	


				//actions

				if (kit_w < 320 - 50) {
					game->makeAction(actions[0]);
					game->makeAction(actions[0]);


				}
				else if (kit_w > 320 + 40) {
					game->makeAction(actions[1]);
					game->makeAction(actions[1]);

				}

				else {
					game->makeAction(actions[3]);
				}
				game->makeAction(actions[3]);

			}

			else {
				game->makeAction(actions[2]);
				game->makeAction(actions[2]);

				game->makeAction(actions[1]);
			}



			cv::imshow("Output Window", greyscale);
			//cv::imshow("Track window", image);

			cv::waitKey(1);

		}


		std::cout << std::endl << game->getTotalReward() << std::endl;
		total_reward += game->getTotalReward();
	}

	std::cout << std::endl << "Total average" << total_reward / 10 << std::endl;
}
	

int main()
{
	game->setViZDoomPath(path + "\\vizdoom.exe");
	game->setDoomGamePath(path + "\\freedoom2.wad");

	//cv::namedWindow("Output Window", cv::WINDOW_AUTOSIZE);
	
	auto episodes = 10;
	RunTask1(episodes);
	
	game->close();

}									
