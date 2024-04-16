import sys
import time
import pygame
import imageio 

save_gif = False

class TurtleStockGraph:
    """
    The StockGraph class is responsible for visualizing stock price movements and corresponding trading actions.
    It creates a graphical representation of stock prices and marks the points of buy and sell actions. Used in
    render function for gym
    """
    def __init__(self, width, height, background_color, dynamic_exit, max_window_size=100, stock_name=None):
        """
        Initializes the StockGraph object.

        Args:
            width (int): The width of the graph window.
            height (int): The height of the graph window.
            background_color (tuple): The background color of the graph in RGB format.
        """

        self.width, self.height = width, height
        self.background_color = background_color
        self.initialized = False
        self.max_window_size = max_window_size
        self.dynamic_exit = dynamic_exit
        self.colors = {
            0: (0, 255, 0),  # Green for 'buy'
            1: (255, 0, 0),  # Red for 'sell'
        }
        self.stock_name = stock_name
        
        
        self.frames = []
        

    def process_events(self):
        """
        Run in thread so you can move the pygame window
        """
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        except:
            time.sleep(1)

    def update_graph(self, prices, actions, rolling_high, rolling_low, directions, date, current_entry, current_exit):
        """
        Updates the graph with prices, actions, rolling highs, and rolling lows.
        It colors the rolling highs based on the action taken.
        
        Blue -> Expand
        Yellow -> Contract
        Magenta -> No change
        
        """
        if len(prices) > self.max_window_size:
            prices = prices[-self.max_window_size:]
            actions = actions[-self.max_window_size:]
            rolling_high = rolling_high[-self.max_window_size:]
            rolling_low = rolling_low[-self.max_window_size:]
            directions = directions[-self.max_window_size:]
        
        
        if not (len(prices) == len(actions) == len(rolling_high) == len(rolling_low) == len(directions)):
            raise ValueError(
                "Length of prices, actions, rolling high, rolling low, and directions must be the same.")

        min_price = min(prices + rolling_high + rolling_low)
        max_price = max(prices + rolling_high + rolling_low)
        if max_price == min_price:
            max_price = min_price + 1  # Avoid division by zero

        self.screen.fill(self.background_color)
    
        title_text = self.font.render(self.stock_name + " Stock Graph", True, (255, 255, 255))
        title_position = ((self.width - title_text.get_width()) // 2, 10)  # Center the title
        self.screen.blit(title_text, title_position)
        
        date_text = self.font.render(date, True, (255, 255, 255))
        date_position = ((self.width - date_text.get_width()) // 2, self.height - 30)  # Center horizontally and near the bottom
        self.screen.blit(date_text, date_position)

        self._draw_axes_and_labels(prices, len(prices))
        self._draw_gridlines(len(prices), min_price, max_price)

        # Plot the price line
        for i in range(1, len(prices)):
            x1, y1 = self._calculate_coordinates(i - 1, prices[i - 1], min_price, max_price, len(prices))
            x2, y2 = self._calculate_coordinates(i, prices[i], min_price, max_price, len(prices))
            pygame.draw.line(self.screen, (255, 255, 255), (x1, y1), (x2, y2))

        # Plot rolling highs
        for i in range(1, len(rolling_high)):
            x1, y1 = self._calculate_coordinates(i - 1, rolling_high[i - 1], min_price, max_price, len(prices))
            x2, y2 = self._calculate_coordinates(i, rolling_high[i], min_price, max_price, len(prices))
            rolling_high_color = self._direction_color(directions[i])
            pygame.draw.line(self.screen, rolling_high_color, (x1, y1), (x2, y2))

        # Plot rolling lows
        for i in range(1, len(rolling_low)):
            x1, y1 = self._calculate_coordinates(i - 1, rolling_low[i - 1], min_price, max_price, len(prices))
            x2, y2 = self._calculate_coordinates(i, rolling_low[i], min_price, max_price, len(prices))
            if self.dynamic_exit:
                rolling_low_color = self._direction_color(directions[i])
            else:
                rolling_low_color = (255, 0, 255)  # Magenta, for static rolling_low color
            pygame.draw.line(self.screen, rolling_low_color, (x1, y1), (x2, y2))

        # Plot buy/sell action points
        for i, (price, action) in enumerate(zip(prices, actions)):
            if action in [0, 1]:  # Assuming these are valid actions
                x, y = self._calculate_coordinates(i, price, min_price, max_price, len(prices))
                color = self.colors[action]
                pygame.draw.circle(self.screen, color, (int(x), int(y)), 5)
                
        entry_text = self.font.render(f"Entry Period: {current_entry}", True, (255, 255, 255))  # Yellow text
        entry_position = (self.width - entry_text.get_width() - 10, 10)  # Top-right corner
        self.screen.blit(entry_text, entry_position)

        # Display current exit period in the bottom right
        exit_text = self.font.render(f"Exit Period: {current_exit}", True, (255, 255, 255))  # Cyan text
        exit_position = (self.width - exit_text.get_width() - 10, self.height - 30)  # Bottom-right corner
        self.screen.blit(exit_text, exit_position)

        self._draw_legend()
        pygame.display.flip()
        
        if save_gif:
            frame_image = pygame.surfarray.array3d(pygame.display.get_surface())
            frame_image = frame_image.transpose([1, 0, 2])
            self.frames.append(frame_image)
            
        
    def save_as_gif(self, filename='output.gif', duration=0.1):
        # Save the captured frames as a GIF
        imageio.mimsave(filename, self.frames, duration=duration)
        
    def _direction_color(self, direction):
        """
        Determines the color based on the direction. Used for helping track the agent actions
        """
        if direction == 0:
            return (0, 0, 255)  # Blue
        elif direction == 1:
            return (255, 255, 0)  # Yellow
        else:
            return (255, 0, 255)  # Magenta

    def _calculate_coordinates(self, index, price, min_price, max_price, total_prices):
        """
        Calculate the graphical coordinates for a given price at a specific index.

        Args:
            index (int): The index of the price in the prices list.
            price (float): The price value to be plotted.
            min_price (float): The minimum price value in the dataset.
            max_price (float): The maximum price value in the dataset.
            total_prices (int): The total number of prices in the dataset.

        Returns:
            tuple: The (x, y) coordinates for plotting.
        """
        x = 50 + index * (self.width - 100) / (total_prices - 1)
        y = self.height - 50 - ((price - min_price) / (max_price - min_price)) * (self.height - 100)
        return x, y

    def close_window(self):
        """
        Closes the Pygame window and terminates the Pygame instance.
        """
        pygame.display.quit()
        pygame.quit()
        
    def _initialize_window(self):
        """
        Initializes the Pygame window and other necessary components.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill(self.background_color)
        self.font = pygame.font.Font(None, 24)

        # Set window title with stock name
        pygame.display.set_caption(self.stock_name + " Stock Graph")

        # Optionally, draw the stock name at the top of the window
        title_surf = self.font.render(self.stock_name + " Stock Graph", True, (255, 255, 255))
        self.screen.blit(title_surf, ((self.width - title_surf.get_width()) // 2, 10))

        pygame.display.flip()
        self.initialized = True


    def _draw_axes_and_labels(self, prices, num_steps):
        """
        Draws the axes and labels for the stock graph.

        Args:
            prices (list): A list of stock prices, used to determine the range of the y-axis.
            num_steps (int): The number of steps (points in time) to label on the x-axis.
        """

        min_price = int(min(prices))
        max_price = int(max(prices))
        if max_price == min_price:
            max_price = min_price + 1  # Avoid division by zero

        # Draw x and y axes
        pygame.draw.line(self.screen, (255, 255, 255), (50, self.height - 50), (self.width - 50, self.height - 50))
        pygame.draw.line(self.screen, (255, 255, 255), (50, 50), (50, self.height - 50))

        # Draw y-axis labels (prices)
        label_step = max(1, (max_price - min_price) // 5)
        for i in range(min_price, max_price + 1, label_step):
            y = self.height - 50 - ((i - min_price) / (max_price - min_price)) * (self.height - 100)
            label = self.font.render(str(i), True, (255, 255, 255))
            self.screen.blit(label, (5, y - label.get_height() // 2))


    def _draw_gridlines(self, num_steps, min_price, max_price):
        """
        Draws gridlines on the stock graph for better readability.

        Args:
            num_steps (int): The number of vertical gridlines to draw, based on the number of time steps.
            min_price (float): The minimum price, used to calculate the spacing of horizontal gridlines.
            max_price (float): The maximum price, used to calculate the spacing of horizontal gridlines.
        """

        max_price = int(max_price)
        min_price = int(min_price)

        if max_price == min_price:
            max_price = min_price + 1  # Avoid division by zero

        # Draw horizontal gridlines
        label_step = max(1, (max_price - min_price) // 5)
        for i in range(min_price, max_price + 1, label_step):
            y = self.height - 50 - ((i - min_price) / (max_price - min_price)) * (self.height - 100)
            pygame.draw.line(self.screen, (50, 50, 50), (50, y), (self.width - 50, y))

        # Draw vertical gridlines
        for i in range(num_steps):
            x = 50 + (i * (self.width - 100) / (num_steps - 1))
            pygame.draw.line(self.screen, (50, 50, 50), (x, 50), (x, self.height - 50))
            
            
    def _draw_legend(self):
        """
        Draws a legend on the graph to explain what the colors and symbols mean:
        Blue -> Expand period
        Yellow -> Contract period
        Magenta -> Do nothing
        Green Circle -> Buy a unit
        Red Circle -> Sell all units
        """
        legend_entries = [
            ("Blue", (0, 0, 255), "Expand period"),
            ("Yellow", (255, 255, 0), "Contract period"),
            ("Magenta", (255, 0, 255), "Do nothing"),
            ("Green Circle", (0, 255, 0), "Buy a unit"),
            ("Red Circle", (255, 0, 0), "Sell all units")
        ]

        start_x = 50  # Starting x coordinate for the legend
        start_y = 20  # Starting y coordinate for the legend

        for i, (name, color, description) in enumerate(legend_entries):
            if "Circle" in name:
                # Draw the circle for buy/sell indicators
                pygame.draw.circle(self.screen, color, (start_x + 10, start_y + i * 30 + 10), 10)
            else:
                # Draw the color box for color-coded lines
                pygame.draw.rect(self.screen, color, (start_x, start_y + i * 30, 20, 20))
            
            # Draw the description text
            text = self.font.render(description, True, (255, 255, 255))
            self.screen.blit(text, (start_x + 25, start_y + i * 30))

        # Optionally, draw a box around the legend
        pygame.draw.rect(self.screen, (255, 255, 255), (start_x - 10, start_y - 10, 280, 180), 1)
